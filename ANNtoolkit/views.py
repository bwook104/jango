from django.shortcuts import render
from django.http import HttpResponse

import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
import numpy as np
import math

# Create your views here.
def main(request):
    return render(request,'ANNtoolkit/main_page.html')

def result(request):

    d = float(request.POST['d'])
    R = float(request.POST['R'])
    rho = float(request.POST['rho'])
    mu = float(request.POST['mu'])
    Q = float(request.POST['Q'])

    tf.reset_default_graph()

    X=tf.placeholder(tf.float32 , shape=[None,2])
    Y=tf.placeholder(tf.float32 , shape=[None,1])

    w1=tf.get_variable("w1", shape=[2,20],initializer=tf.contrib.layers.variance_scaling_initializer())
    b1=tf.Variable(tf.random_uniform([1], minval=0., maxval=1), name='bias1')
    layer1=tf.sigmoid(tf.matmul(X,w1)+b1)


    w5=tf.get_variable("w5", shape=[20,1],initializer=tf.contrib.layers.variance_scaling_initializer())
    b5=tf.Variable(tf.random_uniform([1], minval=0., maxval=1), name='bias5')
    hypo=tf.matmul(layer1,w5)+b5

    cost=tf.reduce_mean(tf.square(hypo-Y))

    saver = tf.train.Saver()

    save_file = 'D:\KBW\8coding\mysite\jango\ANNtoolkit\Trained_ANNmodel/K_90bend.ckpt'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_file)

        Velocity = (4*Q)/(d*d*math.pi)

        V = round(Velocity,2)
 
        Re=(rho*Velocity*d)/mu

        RE = round(Re,2)

        Re_log = math.log10(Re)

        Rr=R/(d/2)

        feed_input = np.array([[Re_log,Rr]])

        prediction = sess.run(hypo, feed_dict={X : feed_input})

        prediction_result = prediction[0,0]
        Pressure_drop=prediction_result*rho*0.5*Velocity*Velocity
        Pa=round(Pressure_drop,2)

    if Velocity <=0.01:
            return render(request, 'ANNtoolkit/result_page.html', {'K':'-1', 'RE':'-1', 'V':'-1', 'Pa':'-1', 'WS': 'The fluid flow is too slow. Prediction is not possible'})

    if Velocity >=1000:
            return render(request, 'ANNtoolkit/result_page.html', {'K':'-1', 'RE':'-1', 'V':'-1', 'Pa':'-1', 'WS': 'The fluid flow is too fast. Prediction isn’t possible'})

    if Velocity > 0.01 and Velocity<1000:
            if prediction_result >=0:
                    return render(request,'ANNtoolkit/result_page.html',{'K':prediction_result, 'RE':RE, 'V':V, 'Pa':Pa, 'WS':'Prediction is complete!'})
    else:
        return render(request, 'ANNtoolkit/result_page.html', {'K':'-1', 'RE':'-1', 'V':'-1', 'Pa':'-1', 'WS': 'Prediction is not possible'})