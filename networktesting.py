# trainingData, validationData, testData = mnistLoader.load(expanded=False, short=False)
# expandedTrainingData, validationData, testData = mnistLoader.load(expanded=True, short=False)
# shortTrainingData, validationData, testData = mnistLoader.load(expanded=False, short=True)
# myimgs = loadMyImages("mytrainimages3.pkl.gz")
# myimgsExp = loadMyImages("mytrainimages3_expanded.pkl.gz")
# mytestimgs = loadMyImages("mytestimages3.pkl.gz")
# mytestimgsExp = loadMyImages("mytestimages3_expanded.pkl.gz")
#
# comboTrainingData = shortTrainingData
# for i in range(10):
#     comboTrainingData = comboTrainingData + myimgsExp
# print("combo mnist and own training set length =", len(comboTrainingData))

# supernetwork = loadNetwork(name="mnist_stand_4000")
# supernetwork2 = loadNetwork(name="mnist_exp_8520")
# supernetwork3 = loadNetwork(name="mnist_short_4780")
# supernetwork4 = loadNetwork(name="mnist20k_myimgs5k", trainingData=comboTrainingData, valiData=mytestimgsExp)
# supernetwork4 = loadNetwork(name="mnist20k_myimgs20k_2960") #trainingData=comboTrainingData, testData=mytestimgsExp)

# ff = open("NetworkTesting2", "w")
# ff.write("100 hidden layers, 30 epochs, minibatch size 10, eta=0.5")
# s = "mnist_stand mnist test: "
# s += str(evaluate(supernetwork, testData))
# s += "\nmnist_stand my test: "
# s += str(evaluate(supernetwork, mytestimgs))
# s += "\nmnist_stand my test exp: "
# s += str(evaluate(supernetwork, mytestimgsExp))
# ff.write(s)
# print()
# s = "\nmnist_exp mnist test: "
# s += str(evaluate(supernetwork2, testData))
# s += "\nmnist_exp my test: "
# s += str(evaluate(supernetwork2, mytestimgs))
# s += "\nmnist_exp my test exp: "
# s += str(evaluate(supernetwork2, mytestimgsExp))
# ff.write(s)
# print()
# s = "\nmnist_short mnist test: "
# s += str(evaluate(supernetwork3, testData))
# s += "\nmnist_short my test: "
# s += str(evaluate(supernetwork3, mytestimgs))
# s += "\nmnist_short my test exp: "
# s += str(evaluate(supernetwork3, mytestimgsExp))
# ff.write(s)
# print()
# s = "\nmnist_20k_myimgs_5k mnist test: "
# s += str(evaluate(supernetwork4, testData))
# s += "\nmnist_20k_myimgs_5k my test: "
# s += str(evaluate(supernetwork4, mytestimgs))
# s += "\nmnist_20k_myimgs_5k my test exp: "
# s += str(evaluate(supernetwork4, mytestimgsExp))
# ff.write(s)
# ff.close()