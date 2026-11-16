# Note de synthèse : CUDA contre OpenMP target sur un PCG Jacobi (GH200)

*ROMEO, nœud romeo-a041, 1 GPU GH200 (HBM3 96 Go), NVHPC 24.11, CUDA 12.6. Jobs SLURM 826985 et 827121 du 16/11/2026 (le second ajoute la fusion de noyaux ; les autres chiffres sont stables à 2 % près) : 5 répétitions par mesure, médianes. Chiffres détaillés dans `results/summary.md`.*

## Ce qu'on mesure

On compare 9 versions du même PCG Jacobi en CSR, sur un Poisson 2D (jusqu'à 16,8 M d'inconnues), un Poisson 3D (jusqu'à 16,8 M) et un Poisson 2D à coefficient variable avec un contraste de 10³. On mesure les noyaux séparément (SpMV, produit scalaire, axpy) et le solveur complet. Le STREAM triad mesuré atteint **3,5 To/s**, soit 87 % du pic théorique de 4 To/s.

## Résultats

1. **Justesse.** Sur les Poisson, toutes les versions convergent en **exactement le même nombre d'itérations** (894 à 6747). Sur le cas à coefficient variable, mal conditionné, le nombre d'itérations varie d'environ 3 % **d'une exécution à l'autre, pour une même version**. La cause : les `atomicAdd` du produit scalaire somment dans un ordre non déterministe, et un CG mal conditionné amplifie ces écarts d'arrondi. Pour ce cas, il faut donc comparer le **temps par itération**. Toutes les solutions vérifient `‖b − Ax‖/‖b‖ < 1e-8` (recalculé sur l'hôte).
2. **OpenMP « par défaut » : 1,55 à 1,85× plus lent que CUDA** sur le CG complet. C'est le cas des versions `omp-naive`, `omp-devptr`, `omp-loop` et `omp-unified`. Cet ordre de grandeur rejoint les « 57 % du natif en moyenne » de Gayatri et al. (Kokkos OpenMPTarget, A100).
3. **OpenMP réglé (`omp-opt`) : 1,05 à 1,12× sur les grands cas** (Poisson 2D 4096², Poisson 3D 256³, coefficient variable 2048², en temps par itération). Le gain vient de **deux causes vérifiées au profileur (`nsys`)** :
   - **Produit scalaire.** Par défaut, NVHPC lance la réduction sur **32 768 blocs** de 128 threads (n/128), et plafonne à environ 25 % du pic. En bornant la grille (`num_teams(132*8)`, comme en CUDA), on lance 1056 blocs et on atteint **76 %**, contre 80 % pour CUDA. C'est **le premier poste de l'écart**, comme dans la littérature : les réductions GPU restent le point faible des runtimes OpenMP.
   - **SpMV.** Sans `thread_limit`, le noyau est compilé avec **89 registres par thread**. Avec `thread_limit`, il en utilise **32**, ce qui donne une meilleure occupation : 99 µs contre 133 µs, et 86 % du pic contre 65 %. Le `thread_limit` joue le rôle d'un `__launch_bounds__`. Sa valeur exacte (64 à 1024) change peu la SpMV ; 256 est le meilleur choix pour le produit scalaire.
4. **axpy : OpenMP rattrape CUDA dès que le vecteur est grand** (86 % contre 91 % du pic à 16,8 M). Sur les petits vecteurs, **le coût de lancement** domine : environ 9 µs par région `target` NVHPC, contre environ 2,4 µs par noyau CUDA. Sur un petit problème, avec 8 lancements par itération, c'est ce qui coûte le plus. Cela rejoint l'observation d'Yvan Fournier : plus l'opération est simple, plus OpenMP est proche de CUDA, à condition que le problème soit assez gros.
5. **Fusion de noyaux** (8 → 5 lancements par itération, moins d'octets lus). Elle fait gagner 10 % à CUDA et 10 à 15 % à OpenMP, qui paie plus cher chaque lancement. `omp-fused` devient **plus rapide que le CUDA non fusionné** (0,93 à 0,96× sur les grands cas), mais reste 4 à 8 % derrière `cuda-fused`. **La structure de l'algorithme compte autant que le modèle de programmation.**
6. **Mapping et `loop`.** `is_device_ptr` contre `target enter data` : **aucune différence mesurable** avec NVHPC, la recherche dans la table de mapping est négligeable. `teams loop` gagne de 5 à 10 % sur la SpMV, mais ne corrige pas la réduction.
7. **Mémoire unifiée du GH200.**
   - `omp-unified` (`-gpu=mem:unified`, simple `malloc`) se comporte comme `omp-devptr`.
   - `cuda-managed` et `cuda-system` (`malloc` lu par le GPU via NVLink-C2C) font **jeu égal avec `cudaMalloc` en 2D** (moins de 4 %), mais perdent **12 à 15 % en Poisson 3D 256³**.
   - Une fois les données résidentes, la mémoire unifiée ne coûte donc presque rien. Je n'ai pas mesuré ce qui se passe page par page : c'est à vérifier avant de généraliser.
8. **cuSPARSE est 15 à 35 % plus lent** que la SpMV CSR écrite à la main (« un thread par ligne ») sur ces matrices à 5 ou 7 non-zéros par ligne. L'algorithme générique ne profite pas de la régularité du stencil. Ce résultat ne dit rien des matrices non structurées.

## Où OpenMP rattrape CUDA, où il reste derrière, pourquoi

- **Il rattrape** : les noyaux simples sur de gros vecteurs (axpy), et la SpMV CSR si l'on fixe `thread_limit`.
- **Il reste derrière** : les **réductions**, tant que la grille n'est pas bornée à la main, et les **petits problèmes**, à cause du coût de lancement des régions `target`.
- **Pourquoi** : les choix par défaut du compilateur (taille de grille, registres) ne sont pas ceux qu'un développeur CUDA ferait. Il faut **les imposer par des clauses**, et le bon réglage dépend du compilateur (c'est la conclusion de Gayatri et al.). Le code OpenMP n'est donc pas « portable en performance » sans réglage par plateforme.

## Limites

Il n'y a pas de LLVM/clang sur ROMEO : les optimisations LLVM de Sala et al. (IWOMP 2025, papier non lu en entier, accès Springer) **ne sont pas testées**. Les résultats valent pour NVHPC 24.11 sur un seul GPU. Il n'y a ni multigrille (voir le projet mg-pcg-gpu), ni format MSR. Sur les petites tailles, les valeurs au-dessus de 100 % du pic viennent du cache L2 (50 Mo).

## Trois questions pour Yvan Fournier

1. Sur code_saturne, avez-vous observé que l'écart OpenMP/CUDA vient surtout des **réductions** (produits scalaires du CG) ? Chez moi, NVHPC lance la réduction sur n/128 blocs, ce qui la limite à 25 % du pic, jusqu'à ce qu'on borne `num_teams`. Vos backends OpenMP imposent-ils des clauses de ce type ?
2. Votre solveur de pression **fusionne-t-il les noyaux** du CG (axpy et produit scalaire, par exemple) ? Avec environ 9 µs par région `target` contre 2,4 µs par noyau CUDA, la fusion devrait profiter davantage à OpenMP, surtout sur les niveaux grossiers du multigrille où les vecteurs sont petits.
3. Sur GH200, utilisez-vous la **mémoire unifiée** (`malloc` simple) en production ? Chez moi, elle ne coûte rien en 2D mais 12 à 15 % sur un Poisson 3D 256³. Est-ce que cela correspond à ce que vous voyez sur des maillages non structurés ?
