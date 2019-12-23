import numpy
import scipy.special
import matplotlib.pyplot
import random
import scipy.ndimage as ndi

class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes1, hiddennodes2,outputnodes,learningrate):

        self.inodes=inputnodes
        self.hnodes1=hiddennodes1
        self.hnodes2=hiddennodes2
        self.onodes=outputnodes
        

        '''
        numpy.random.normal(평균, 표준편차, 숫자) : 평균과 표준편차를 가지는 숫자개의 Gaussian distribution 랜덤 넘버를 생성
        numpy.random.normal(평균, 표준편차, (row, column)) : 평균과 표준편차를 가지는 랜덤 넘버를 가지는 row*column 행렬 생성(정확히는 2차원 array)
        '''
        self.wih=numpy.random.normal(0.0,pow(self.hnodes1,-0.5),(self.hnodes1,self.inodes))
        self.whh=numpy.random.normal(0.0,pow(self.hnodes2,-0.5),(self.hnodes2,self.hnodes1))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes2))
        self.lr=learningrate

        self.activation_function=lambda x: scipy.special.expit(x)
        self.inverse_activation_fuction=lambda x: scipy.special.logit(x)

        pass

    def train(self, inputs_list, targets_list):

        inputs=numpy.array(inputs_list, ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        hidden_inputs1=numpy.dot(self.wih, inputs)

        hidden_outputs1=self.activation_function(hidden_inputs1)

        hidden_inputs2=numpy.dot(self.whh, hidden_outputs1)

        hidden_outputs2=self.activation_function(hidden_inputs2)

        final_inputs=numpy.dot(self.who, hidden_outputs2)

        final_outputs=self.activation_function(final_inputs)

        output_errors=targets-final_outputs

        hidden_errors2=numpy.dot(self.who.T, output_errors)

        hidden_errors1=numpy.dot(self.whh.T, hidden_errors2)

        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs2))

        self.whh+=self.lr*numpy.dot((hidden_errors2*hidden_outputs2*(1.0-hidden_outputs2)), numpy.transpose(hidden_outputs1))

        self.wih+=self.lr*numpy.dot((hidden_errors1*hidden_outputs1*(1.0-hidden_outputs1)), numpy.transpose(inputs))

        pass


    def query(self, inputs_list):

        inputs=numpy.array(inputs_list, ndmin=2).T

        hidden_inputs1=numpy.dot(self.wih, inputs)

        hidden_outputs1=self.activation_function(hidden_inputs1)

        hidden_inputs2=numpy.dot(self.whh, hidden_outputs1)

        hidden_outputs2=self.activation_function(hidden_inputs2)

        final_inputs=numpy.dot(self.who, hidden_outputs2)

        final_outputs=self.activation_function(final_inputs)

        return final_outputs

    #학습한 후의 weight를 weights.txt에 저장하는 메서드
    def writeWeights(self):
        fwrite=open("weights.txt",'w')
        
        #write weights between input node and first hidden node
        for num in self.wih:
            fwrite.write(num+' ')
        fwrite.write('\n')

        #write weights between first hidden node and second hidden node
        for num in self.whh:
            fwrite.write(num+' ')
        fwrite.write('\n')

        #write weights between second hidden node and output node
        for num in self.who:
            fwrite.write(num+' ')
        fwrite.write('\n')

        fwrite.close()
    
    #weights.txt에 있는 weights들을 읽어내어 neural network에 저장하는 메서드
    def readWeights(self):
        fread=open("weights.txt", 'r')

        #read first line and set wih
        str_wih=fread.readline()
        self.wih=numpy.asfarray(float(str_wih.split()))

        #read second line and set whh
        str_whh=fread.readline()
        self.whh=numpy.asfarray(str_whh.split())

        #read third line and who
        str_who=fread.readline()
        self.who=numpy.asfarray(str_who.split())

        fread.close()

    def backquery(self, targets_list):

        final_outputs=numpy.array(targets_list, ndmin=2).T

        final_inputs=self.inverse_activation_fuction(final_outputs)

        hidden_outputs2=numpy.dot(self.who.T, final_inputs)

        hidden_outputs2-=numpy.min(hidden_outputs2)
        hidden_outputs2/=numpy.max(hidden_outputs2)
        hidden_outputs2*=0.98
        hidden_outputs2+=0.01

        hidden_inputs2=self.inverse_activation_fuction(hidden_outputs2)

        hidden_outputs1=numpy.dot(self.whh.T, hidden_inputs2)

        hidden_outputs1-=numpy.min(hidden_outputs1)
        hidden_outputs1/=numpy.max(hidden_outputs1)
        hidden_outputs1*=0.98
        hidden_outputs1+=0.01

        hidden_inputs1=self.inverse_activation_fuction(hidden_outputs1)

        inputs=numpy.dot(self.wih.T, hidden_inputs1)

        inputs-=numpy.min(inputs)
        inputs/=numpy.max(inputs)
        inputs*=0.98
        inputs+=0.01

        return inputs
    
input_nodes=784 #28*28 pixel
hidden_nodes=302
hidden_nodes2=126
output_nodes=10 #0~9

learning_rate=0.1

performList=list()

n=neuralNetwork(input_nodes, hidden_nodes, hidden_nodes2,  output_nodes, learning_rate)

training_data_file=open("./mnist_train.csv","r")

training_data_list=training_data_file.readlines()
training_data_file.close()

test_data_file=open("./all_custom_data.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

xlist=list(range(20))

for trynum in xlist:

    for record in training_data_list:

        all_values=record.split(',')

        #asfarray : float 형태의 데이터를 가지는 배열을 만들 때.
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.98)+0.01

        targets=numpy.zeros(output_nodes)+0.01

        targets[int(all_values[0])]=0.99
        n.train(inputs, targets)
        
        #data augmentation

        #색반전
        new_inputs=list()
        [new_inputs.append(255.0-float(num)) for num in all_values[1:]]
        new_arr=(numpy.asfarray(new_inputs)/255.0*0.98)+0.01
        n.train(new_inputs, targets)

        #random rotation
        new_inputs=inputs.reshape(28,28)
        #center of input and output
        c_in=0.5*numpy.array((28,28))
        c_out=0.5*numpy.array((28,28))

        #angle limit
        rg=60.0
        theta=numpy.pi/180*numpy.random.uniform(-rg,rg)
        rotation_matrix=numpy.array([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]])
        inv_rotation=rotation_matrix.T
        offset=c_in-numpy.dot(inv_rotation,c_out)

        out=(ndi.interpolation.affine_transform(new_inputs,inv_rotation,order=2,offset=offset,output=numpy.float32,mode="nearest"))
        newout=out.flatten()
        n.train(newout,targets)



        #random zoom
        new_inputs=inputs.reshape(28,28)
        #zoom limit
        z1=1.5; z2=2.0
        zx,zy=numpy.random.uniform(z1,z2,2)
        inv_zoom_matrix=numpy.diag([1/zx, 1/zy])
        offset=c_in-numpy.dot(inv_zoom_matrix, c_out)

        out=(ndi.interpolation.affine_transform(new_inputs, inv_zoom_matrix, order=2, offset=offset, output=numpy.float32, mode="nearest"))
        newout=out.flatten()
        n.train(newout,targets)



        #random shear
        new_inputs=inputs.reshape(28,28)

        #intensity limit
        intensity=30.0

        theta=numpy.pi/180*numpy.random.uniform(-intensity, intensity)
        inv_rotation=numpy.array([[numpy.cos(theta),numpy.sin(theta)],[0,1]]/numpy.cos(theta))
        offset=c_in-numpy.dot(inv_rotation, c_out)

        out=(ndi.interpolation.affine_transform(new_inputs, inv_rotation, order=2, offset=offset, output=numpy.float32, mode="nearest"))
        newout=out.flatten()
        n.train(newout, targets)



        #random shift
        new_inputs=inputs
        shift=random.randint(1,3)    #1pixel~3pixel중 랜덤한 픽셀로 x축 방향으로 shift
        for i in range(28):
            for j in range(28):
                if j>=28-shift:
                    continue
                #data shift
                new_inputs[i*28+j+shift]=new_inputs[i*28+j]
        n.train(new_inputs, targets)

        new_inputs=inputs
        shift=random.randint(1,3)    #1pixel~3pixel중 랜덤한 픽셀로 y축 방향으로 shift
        for i in range(28):
            for j in range(28):
                if i>=28-shift:
                    continue
                #data shift
                new_inputs[(i+shift)*28+j]=new_inputs[i*28+j]
        n.train(new_inputs, targets)

        new_inputs=inputs
        shift=random.randint(1,3)    #1pixel~3pixel중 랜덤한 픽셀로 -y축 방향으로 shift
        for i in range(28):
            for j in range(28):
                if i<=shift:
                    continue
                #data shift
                new_inputs[(i-shift)*28+j]=new_inputs[i*28+j]
        n.train(new_inputs, targets)

        new_inputs=inputs
        shift=random.randint(1,3)    #1pixel~3pixel중 랜덤한 픽셀로 -x축 방향으로 shift
        for i in range(28):
            for j in range(28):
                if j<=shift:
                    continue
                #data shift
                new_inputs[i*28+j-shift]=new_inputs[i*28+j]
        #Early Stopping용 코드
        if n.train(new_inputs, targets):
            flag=True

        pass

    pass

scorecard=[]

for record in test_data_list:

    all_values=record.split(",")

    correct_label=int(all_values[0])

    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01

    outputs=n.query(inputs)

    label=numpy.argmax(outputs)
    print(label,correct_label)
    if label==correct_label:

        scorecard.append(1)
    else:

        scorecard.append(0)
        pass
    
    pass

scorecard_array=numpy.asarray(scorecard)
performance=scorecard_array.sum()/scorecard_array.size
print("\nPerformance = ", performance)