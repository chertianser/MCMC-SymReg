from datasets import load_dataset
import pandas as pd
from bsr.bsr_class import BSR
from pysr import PySRRegressor

def get_srsd_set(dataset,equation_number, fold='train'):
    fold_multiplier = 8000 if fold == 'train' else 1000
        
    pandas_fold = dataset[fold].to_pandas()
    row_number = equation_number*fold_multiplier

    tmp = pandas_fold[row_number:row_number+fold_multiplier].pop('text').str.strip('[]').str.split(expand=True).astype(float)
    X_train = tmp.iloc[:,0:-1]
    y_train = tmp.iloc[:,-1]
    return X_train, y_train

run_bsr = True
run_pysr = False
srsd = False
bsr_dataset = True


if srsd:
    dataset = load_dataset("yoshitomo-matsubara/srsd-feynman_easy")
    X_train, y_train = get_srsd_set(dataset=dataset, equation_number=0, fold='train')
elif bsr_dataset:
    random.seed(1)
    n = 100
    x1 = np.random.uniform(-3, 3, n)
    x2 = np.random.uniform(-3, 3, n)
    x1 = pd.DataFrame(x1)
    x2 = pd.DataFrame(x2)
    train_data = pd.concat([x1, x2], axis=1)
    train_y = 1.35* train_data.iloc[:,0]*train_data.iloc[:,1] + 5.5*np.sin((train_data.iloc[:,0]-1)*(train_data.iloc[:,1]-1))

    xx1 = np.random.uniform(-3, 3, 30)
    xx2 = np.random.uniform(-3,3,30)
    xx1 = pd.DataFrame(xx1)
    xx2 = pd.DataFrame(xx2)
    test_data = pd.concat([xx1, xx2], axis=1)
    test_y = 1.35* test_data.iloc[:,0]*test_data.iloc[:,1] + 5.5*np.sin((test_data.iloc[:,0]-1)*(test_data.iloc[:,1]-1))

    xxx1 = np.random.uniform(-6, 6, 30)
    xxx2 = np.random.uniform(-6,6,30)
    xxx1 = pd.DataFrame(xx1)
    xxx2 = pd.DataFrame(xx2)
    test2_data = pd.concat([xx1, xx2], axis=1)
    test2_y = 1.35* test2_data.iloc[:,0]*test2_data.iloc[:,1] + 5.5*np.sin((test2_data.iloc[:,0]-1)*(test2_data.iloc[:,1]-1))
    
    X_train = train_data
    y_train = train_y
else
    exit("Specify training set")

if run_bsr:
    params = {'treeNum': 3, 'itrNum':50, 'alpha1':0.4, 'alpha2':0.4, 'beta':-1, 'disp': False, 'val': 100}

    my_bsr = BSR(**params)
    my_bsr.fit(X_train, y_train)

    fitted_y = my_bsr.predict(X_train)
    print(my_bsr.model())
    print(my_bsr.complexity())
if run_pysr:
    model = PySRRegressor(
        niterations=40,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "inv(x) = 1/x",
            "log", # ln
            "neg",
            "sin",
            "cos",
            "exp",
            "square",
            "cube",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )

    model.fit(X_train, y_train)
    print(model)
