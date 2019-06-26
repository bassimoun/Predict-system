import numpy as np
import pickle
import sys


ALBUMIN_IN_URIINE=["NEGATIVE","POSITIVE"]
MATERNAL_PELVIS=["DEFORMITY","NORMAL","SMALL"]
AMNIOTIC_FLUID=["ABNORMAL","NORMAL","SMALL AMOUNT"]
ECLAMPSIA=["ABNORMAL","COMPLICATION","NORMAL"]
FETAL_ASPHYXIA=["ABNORMAL","ASPHYXIA","NORMAL"]
FETAL_HYPOXIA=["ABNORMAL","Neg","NEGATIPos","Pos"]
UMBILICAL_CORD=["ABNORMAL","NORMAL"]
PLACENTA_PREVIA=["ABNORMAL","NORMAL"]
FETAL_POSITION =["ABNORMAL","NORMAL"]
UTERUINE = ["ABNORMAL","ACUTE SITUATION","NORMAL"]

test_input = np.array(((float(sys.argv[1])-8.0)/7.0), dtype=np.float32)
test_input = np.hstack((test_input, np.array(((float(sys.argv[2])-60.0)/500.0), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(ALBUMIN_IN_URIINE.index(sys.argv[3])/(len(ALBUMIN_IN_URIINE)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(MATERNAL_PELVIS.index(sys.argv[4])/(len(MATERNAL_PELVIS)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(AMNIOTIC_FLUID.index(sys.argv[5])/(len(AMNIOTIC_FLUID)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(ECLAMPSIA.index(sys.argv[6])/(len(ECLAMPSIA)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(FETAL_ASPHYXIA.index(sys.argv[7])/(len(FETAL_ASPHYXIA)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(FETAL_HYPOXIA.index(sys.argv[8])/(len(FETAL_HYPOXIA)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(UMBILICAL_CORD.index(sys.argv[9])/(len(UMBILICAL_CORD)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(PLACENTA_PREVIA.index(sys.argv[10])/(len(PLACENTA_PREVIA)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(FETAL_POSITION.index(sys.argv[11])/(len(FETAL_POSITION)-1)), dtype=np.float32)))
test_input = np.hstack((test_input, np.array(float(UTERUINE.index(sys.argv[12])/(len(UTERUINE)-1)), dtype=np.float32)))
test_input = np.vstack((test_input,test_input))
# load the model from disk
loaded_model = pickle.load(open('SVM_model_linear.sav', 'rb'))
y_pred = loaded_model.predict(test_input)
print(y_pred)