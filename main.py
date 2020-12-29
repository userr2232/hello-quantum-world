# Basic tutorial from Xanadu AI: https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html

import pennylane as qml
from pennylane import numpy as np # it is important to import numpy provided by pennylane

# CREATING A DEVICE

# A device is any computational object that can apply quantum operations
# and return a measurement value

dev1 = qml.device("default.qubit", wires=1)

# CONSTRUCTING THE QNODE

# QNodes are an abstract encapsulation of a quantum function, described by a quantum circuit
# we convert our quantum function into a QNode running on dev1 by applying the decodrator qnode
@qml.qnode(dev1)
def circuit(params):
    # params may be a list, tuple or array. It uses the individual elements for gate parameters
    qml.RX(params[0], wires=0) # X rotation
    qml.RY(params[1], wires=0) # Y rotation
    return qml.expval(qml.PauliZ(0)) # measurement

print(circuit([0.54, 0.12]))

# CALCULATING THE QUANTUM GRADIENTS

# We can differentiate using the grad function. 
# This returns another function representing the gradient
# this represents the derivative of the QNode with respect to the argument specified in argnum
dcircuit = qml.grad(circuit, argnum=0)

# evaluate the gradient
print(dcircuit([0.54, 0.12]))

## we could have defined the circuit above with two arguments instead of an array of arguments:
@qml.qnode(dev1)
def circuit2(phi1, phi2):
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
    return qml.expval(qml.PauliZ(0))

dcircuit2 = qml.grad(circuit2, argnum=[0, 1])
print(dcircuit2(0.54, 0.12))

# OPTIMIZATION

# if using NumPy/Autograd interface, PennyLane offers a number of optimizers based on GD
# Now we will use one of PennyLane's optimizers to optimize the two params phi1 and phi2 
# such that the qubit originally in state |0>, is rotated to state |1>

# But first, we need to define a cost function
# Since we need to minimize the cost function and, our desired value is -1 and, the Pauli-Z expectation is bound to [-1, 1],
# we can define our cost as the output of our circuit.

def cost(x):
    return circuit(x)

# we define some initial params to begin our optimization:

init_params = np.array([0.011, 0.012])
print(cost(init_params)) # this will print a value close to 1

# Now we will finally use an optimizer. (100 steps)

opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100

params = init_params

for i in range(steps):
    params = opt.step(cost, params)
    if (i + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))

    print("Optimized rotation angles: {}".format(params))