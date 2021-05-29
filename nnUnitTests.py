from nnFunctions import *
import unittest

# unit tests for nnFunctions
class TestNN(unittest.TestCase):

    def setUp(self):
        self.train_data = np.array([np.linspace(0,1,num=2),np.linspace(1,0,num=2)]).T
        self.train_label = np.array([0,1])
        self.args = (2,2,2, self.train_data, self.train_label, 0)
        self.args1 = (2,2,2, self.train_data, self.train_label, 1)
        self.test_data = np.array([np.linspace(2,16,num=20),np.linspace(-8,12,num=20)]).T
        self.params = np.linspace(-5,5, num=12)
        self.W1 = np.array([[0.2,0.4,0.1],[0.6,0.1,0.4]])
        self.W2 = np.array([[0.4,0.8,0.1],[0.6,0.3,0.5]])

    def test_sigmoid_scalar(self):
        self.assertAlmostEqual(sigmoid(2),0.8807970779778823)

    def test_sigmoid_vector(self):
        self.assertAlmostEqual(np.linalg.norm(sigmoid(np.array([4,3,1])) -
                                              np.array([0.98201379, 0.95257413, 0.73105858])),0)
    def test_sigmoid_matrix(self):
        self.assertAlmostEqual(np.linalg.norm(sigmoid(np.array([[4,2],[3,1]])) -
                                              np.array([[0.98201379, 0.88079708],
                                                        [0.95257413, 0.73105858]])),0)
    def test_nnobjval_obj(self):
        objval,objgrad = nnObjFunction(self.params, *self.args)
        self.assertAlmostEqual(objval,4.055656624827229)

    def test_nnobjval_gradient(self):
        objval,objgrad = nnObjFunction(self.params, *self.args)
        self.assertAlmostEqual(np.linalg.norm(objgrad-np.array([
            5.57158262e-05,1.08636910e-03,1.14208493e-03,3.52693510e-02,
            2.38403259e-01,2.73672610e-01,1.00492921e-04,2.25568731e-02,
            4.17497794e-01,3.44826771e-04,6.94020615e-02,4.95503752e-01
        ])),0)

    def test_nnobjval_obj_reg(self):
        objval,objgrad = nnObjFunction(self.params, *self.args1)
        self.assertAlmostEqual(objval,33.601111170281776)

    def test_nnobjval_gradient_reg(self):
        objval,objgrad = nnObjFunction(self.params, *self.args1)
        self.assertAlmostEqual(np.linalg.norm(objgrad-np.array([
            -2.49994428,-2.04436818,-1.58976701,
            -1.10109429,-0.44341492,0.04639988,
             0.22737322,0.70437505,1.55386143,
            1.59125392 ,2.11485661 ,2.99550375
        ])),0)

    def test_nnpredict(self):
        preds = nnPredict(self.W1, self.W2, self.test_data)
        self.assertAlmostEqual(len(np.where(preds==0)[0]),2)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=3, exit=False)


