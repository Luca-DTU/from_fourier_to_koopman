"""
This is my exploration of the method
- simple artificial data
- fluid flow data
- Energy consumption data
- Solar energy data
- enso data
- Kaggle data
gameplan:
- Fully understand the theory and algorithm

"""
import numpy as np
from fourier_koopman import fourier, koopman, fully_connected_mse
import matplotlib.pyplot as plt
import pandas as pd
class data:
    def artificial(size,split_ratio,noise = True):
        x = (np.sin([2*np.pi/24*np.arange(size)]) + np.sin([2*np.pi/33*np.arange(size)])).T
        if noise:
            x += np.random.normal(0,0.2,x.shape)
        split = int(x.shape[0]*split_ratio)
        x_train = x[:split]
        x_test = x[split:]
        return x_train,x_test,split
    def artificial_2(size,split_ratio):
        f1 = 1.0  # frequency of first sinusoid
        f2 = 1.1  # frequency of second sinusoid
        A1 = 1.1  # amplitude of first sinusoid
        A2 = 0.9  # amplitude of second sinusoid
        phi1 = 0  # phase of first sinusoid
        phi2 = np.pi/2  # phase of second sinusoid
        t_start = 0  # start time
        t_end = 100  # end time
        dt = (t_end-t_start)/size # time step

        # Create time array
        t = np.arange(t_start, t_end, dt)
        # Create signal as superposition of sinusoids
        x = A1 * np.cos(2*np.pi*f1*t + phi1) + A2 * np.cos(2*np.pi*f2*t + phi2)
        x = x.reshape(size,1)
        split = int(x.shape[0]*split_ratio)
        x_train = x[:split]
        x_test = x[split:]
        return x_train,x_test,split
    def lorenz():
        pass
    def ks_equation():
        pass
    def fluid_flow():
        pass
    def energy_consumption(split_ratio = 0.7):
        # https://www.terna.it/en/electric-system/transparency-report/download-center
        df = pd.read_excel('data\Terna\data.xlsx').dropna()
        for i in range(1,6):
            dfi = pd.read_excel(f"data\Terna\data ({i}).xlsx").dropna()
            df = pd.concat([df,dfi])
        df = df.reset_index(drop=True)
        x = df["Total Load [MW]"].values
        x = x.reshape(x.shape[0],1)
        split = int(x.shape[0]*split_ratio)
        x_train = x[:split]
        x_test = x[split:]
        return x_train,x_test,split
         
    def solar_energy():
        pass
    def enso():
        pass
    def kaggle():
        pass

    

if __name__ == '__main__':
    
    split_ratio = 0.75
    x_train,x_test,split = data.artificial_2(5000,split_ratio,)
    # x_train,x_test,split = data.energy_consumption(split_ratio)
    size = x_train.shape[0] + x_test.shape[0]
    plt.figure(figsize = (20,5))
    plt.plot(np.arange(split),x_train)
    plt.plot(np.arange(split,size),x_test)
    plt.show()
    f = fourier(num_freqs=2)
    # f.fit(f.scale(x_train), iterations = 500,verbose = True)
    # xhat_fourier = f.predict(size)
    # xhat_fourier = f.descale(xhat_fourier)
    f.fit(x_train, iterations = 1500,verbose = True)
    xhat_fourier = f.predict(size)
    plt.figure(figsize = (20,5))
    plt.plot(np.arange(size),xhat_fourier,label = 'fourier')
    plt.plot(np.arange(split),x_train,label = 'train')
    plt.plot(np.arange(split,size),x_test,label = 'test')
    plt.legend()
    plt.show()
    # residuals
    plt.figure(figsize = (20,5))
    plt.plot(np.arange(split),x_train-xhat_fourier[:split],label = 'train')
    plt.plot(np.arange(split,size),x_test-xhat_fourier[split:],label = 'test')
    plt.legend()
    plt.show()
    # koopman
    model_object = fully_connected_mse(x_dim=1, num_freqs=2, n=512)
    k = koopman(model_object, device='cpu')
    k.fit(x_train, iterations = 300, interval = 25, verbose=True)
    xhat_koopman = k.predict(size)
    plt.figure(figsize = (20,5))
    plt.plot(np.arange(size),xhat_koopman,label = 'koopman')
    plt.plot(np.arange(split),x_train,label = 'train')
    plt.plot(np.arange(split,size),x_test,label = 'test')
    plt.legend()
    plt.show()
    # residuals
    plt.figure(figsize = (20,5))
    plt.plot(np.arange(split),x_train-xhat_koopman[:split],label = 'train')
    plt.plot(np.arange(split,size),x_test-xhat_koopman[split:],label = 'test')
    plt.legend()
    plt.show()
    # compare
    plt.figure(figsize = (20,5))
    plt.plot(np.arange(size),xhat_fourier,label = 'fourier')
    plt.plot(np.arange(size),xhat_koopman,label = 'koopman')
    plt.plot(np.arange(split),x_train,label = 'train')
    plt.plot(np.arange(split,size),x_test,label = 'test')
    plt.legend()
    plt.show()




