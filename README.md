# ReinforcementLearningPolicyOptimization

Libreria C++20 per l'ottimizzazione black-box di policy neurali.

## Demo con Docker

```bash
# 1. clone 

git clone https://github.com/SepeFr/ReinformentLearningPolicyOptimization.git
cd ReinformentLearningPolicyOptimization

# 2. set-up docker
docker build -t reinforcement-learning-policy-optimization .

# 3. Run a small demo  
docker run --rm reinforcement-learning-policy-optimization
```

## Esempi di model selection con Python

È possibile eseguire una model selection con Docker:

```bash
mkdir -p build/experiments
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/build/experiments:/app/build/experiments" \
  reinforcement-learning-policy-optimization \
  python3 exps/run_experiments.py CARTPOLE CMA_ES MAXIMUM_ITERATIONS
```

Le combinazioni disponibili sono mostrate con:

```bash
docker run --rm reinforcement-learning-policy-optimization \
  python3 exps/run_experiments.py --help
```

## Documentazione

La documentazione Doxygen è disponibile su
<https://sepefr.github.io/ReinformentLearningPolicyOptimization/>.

La copia locale è disponibile in `build/doxygen/html/index.html`.
Per rigenerarla:

```bash
doxygen Doxyfile
```
