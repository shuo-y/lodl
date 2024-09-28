## Code for "Searching for a Loss Function for Decision Focus Learning"

This code used domain problems from [Shah et al.](https://github.com/sanketkshah/LODLs) and [Zharmagambetov et al.](https://github.com/facebookresearch/LANCER). It tests searching for parameters of different predefined types of loss functions for learning a decision aware model. The loss functions are defined in `losses.py`. Available loss functions are weighted squared error, directed weighted squared error, and instance-based weighted squared error.

To run the domain problem of shortest path with weighted squared error

```
 python3 smacdirspcv.py  --n-trials 100  --search-method wmse  --sp-grid="(5, 5)"
```

To adjust the number of search iterations change `n-trials`. Change `wmse` to `mse++` or `idx` for testing directed weighted squared error and instance-based weighted squared error respectively. Setting `sp-grid` for the graph structure for testing.

To run the domain problem of web advertising with weighted squared error

```
 python3 searchbudget.py  --n-trials 100 --search-method wmse  --n-budget 2 --n-fake-targets 500
```

Change the `n-budget` for number of budgets during the web adverting. Setting `n-fake-tagets` for controlling synthesized features for the target.

To run the domain problem of portfolio optimization

```
python3 smacdirpfv4.py --n-trials 100  --search-method wmse
```

The code will compare the performance of two-stage XGBoost tree and another XGBoost model trained by the searched loss function. The decision quality of the above two types of model is printed out.


