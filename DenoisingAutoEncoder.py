import numpy
import random

class DenoisingAutoEncoder:
    def __init__(self,V,H,batchSize=3):
        self.V = V
        self.H = H
        self.batchSize = batchSize
        self.weight = numpy.random.randn(H,V)/10
        self.visBias = numpy.random.randn(H)/10
        self.hidBias = numpy.random.randn(V)/10
        self.clearDelta()

    def clearDelta(self):
        self.saved = 0
        self.sumDeltaWeight = numpy.zeros(self.H*self.V).reshape(self.H,self.V)
        self.sumDeltaVisBias = numpy.zeros(self.H)
        self.sumDeltaHidBias = numpy.zeros(self.V)

    def sigmoid(self,x):
        return 1.0/(1.0+numpy.exp(-x))

    def noising(self,vis):
        return vis+numpy.random.randn(self.V)/10

    def train(self,vis,alpha=0.1):
        hid = self.encode(self.noising(vis))
        out = self.decode(hid)

        deltaWeight = numpy.dot((numpy.dot(self.weight,vis-out)*hid*(1-hid)).reshape(self.H,1),self.noising(vis).reshape(1,self.V))
        deltaWeight = deltaWeight + numpy.dot((vis-out).T.reshape(self.V,1),hid.reshape(1,self.H)).T

        deltaVisBias = numpy.dot(self.weight,vis-out)*hid*(1-hid)

        deltaHidBias = vis-out

        self.saved += 1
        self.sumDeltaWeight += deltaWeight
        self.sumDeltaVisBias += deltaVisBias
        self.sumDeltaHidBias += deltaHidBias
        if self.saved==self.batchSize:
            self.weight += alpha*self.sumDeltaWeight/self.batchSize
            self.visBias += alpha*self.sumDeltaVisBias/self.batchSize
            self.hidBias += alpha*self.sumDeltaHidBias/self.batchSize
            self.clearDelta()

    def encode(self,vis):
        return self.sigmoid(numpy.dot(self.weight,vis)+self.visBias)

    def decode(self,hid):
        return self.sigmoid(numpy.dot(self.weight.T,hid)+self.hidBias)

if __name__=='__main__':
    N = 100
    D = 20
    samples = [numpy.random.random(D) for _ in xrange(D)]
    print samples
    dae = DenoisingAutoEncoder(D,50)
    for epoch in xrange(10000):
        error = 0
        for s in samples:
            dae.train(s)
            error += numpy.sum(numpy.abs(s-dae.decode(dae.encode(s))))
            if epoch % 100 == 0:
                print dae.encode(s)
        if epoch % 100 == 0:
            print error
        random.shuffle(samples)