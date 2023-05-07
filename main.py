
import numpy as np
from fourier_koopman import fourier, koopman, fully_connected_mse
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
np.random.seed(0)

class data:
    def artificial(size,split_ratio,nonlinear = False, noise = True):
        x = (np.sin([2*np.pi/24*np.arange(size)]) + np.sin([2*np.pi/33*np.arange(size)])).T
        if nonlinear:
            x = np.where(x >= 0, np.sqrt(x), -np.sqrt(-x))
        if noise:
            x += np.random.normal(0,0.2,x.shape)
        split = int(x.shape[0]*split_ratio)
        x_train = x[:split]
        x_test = x[split:]
        return x_train,x_test,split
    def artificial_bivariate(size,split_ratio,noise = True):
        x1 = (np.sin([2*np.pi/24*np.arange(size)]) + np.sin([2*np.pi/33*np.arange(size)])).T
        x2 = (np.cos([2*np.pi/24*np.arange(size)]) + np.cos([2*np.pi/33*np.arange(size)])).T
        x = np.concatenate([x1,x2],-1)
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
    def lorenz(split_ratio):
        # Lorenz system parameters
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        # Lorenz system function
        def lorenz_ode(t, xyz):
            x, y, z = xyz
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - beta * z
            return [dxdt, dydt, dzdt]

        # Initial conditions
        x0, y0, z0 = [1.0, 1.0, 1.0]
        xyz0 = [x0, y0, z0]

        # Time span for simulation
        t_span = [0, 150]

        # Solve ODE using scipy.integrate.solve_ivp
        sol = solve_ivp(lorenz_ode, t_span, xyz0,t_eval=np.linspace(0,50,10000))

        x = sol.y.T
        split = int(x.shape[0]*split_ratio)
        x_train = x[:split]
        x_test = x[split:]
        return x_train,x_test,split


    def energy_consumption(split_ratio = 0.7):
        # https://www.terna.it/en/electric-system/transparency-report/download-center
        try:
            df = pd.read_csv("data\Terna\data.csv")
        except FileNotFoundError:
            print("full file not found, parsing individual files")
            df = pd.read_excel('data\Terna\data.xlsx').dropna()
            for i in range(1,6):
                dfi = pd.read_excel(f"data\Terna\data ({i}).xlsx").dropna()
                df = pd.concat([df,dfi])
            df = df.reset_index(drop=True)
            # group to hourly
            df.set_index("Date",inplace = True)
            df = df.groupby(pd.Grouper(freq='H')).mean()
            df.fillna(method = "ffill",inplace = True)
            df.to_csv("data\Terna\data.csv")

        x = df["Total Load [MW]"].values
        x = x.reshape(x.shape[0],1)
        split = int(x.shape[0]*split_ratio)
        x_train = x[:split]
        x_test = x[split:]
        return x_train,x_test,split
         

def plot(x_train,x_test,xhat_fourier,xhat_koopman,split,size):
    fig,axs = plt.subplots(2,2,figsize = (12,6),sharex='col', sharey='row')
    axs[0,0].plot(np.arange(split),x_train,color = 'orange')
    axs[0,0].plot(np.arange(split,size),x_test,color = 'green')
    axs[0,0].plot(np.arange(size),xhat_koopman,label = 'fit',linestyle = '--',color = 'black')
    axs[0,0].legend(loc = 'lower left')
    axs[0,0].set_title('Koopman')
    axs[0,1].plot(np.arange(split),x_train,color = 'orange')
    axs[0,1].plot(np.arange(split,size),x_test,color = 'green')
    axs[0,1].plot(np.arange(size),xhat_fourier,label = 'fit',linestyle = '--',color = 'black')
    axs[0,1].legend(loc = 'lower left')
    axs[0,1].set_title('Fourier')    
    axs[1,0].plot(np.arange(split),x_train-xhat_koopman[:split],label = 'train',color = 'orange')
    axs[1,0].plot(np.arange(split,size),x_test-xhat_koopman[split:],label = 'test',color = 'green')
    axs[1,0].legend(loc = 'upper left')
    axs[1,0].set_title('Koopman Residuals')
    axs[1,1].plot(np.arange(split),x_train-xhat_fourier[:split],label = 'train',color = 'orange')
    axs[1,1].plot(np.arange(split,size),x_test-xhat_fourier[split:],label = 'test',color = 'green')
    axs[1,1].legend(loc = 'upper left')
    axs[1,1].set_title('Fourier Residuals')
    axs[1,1].set_xlabel('Time')
    axs[0,0].set_ylabel('y')
    axs[1,0].set_ylabel('y')
    axs[1,0].set_xlabel('Time')
    return fig,axs

def main(data,data_kwargs,normalize,num_freqs_fourier,num_freqs_koopman,n_neurons,n_layers,fit_kwargs,figname,sample_num,zoom= ()):
    x_train,x_test,split = data(**data_kwargs)
    size = x_train.shape[0] + x_test.shape[0]
    # plt.figure(figsize = (20,5))
    # plt.plot(np.arange(split),x_train)
    # plt.plot(np.arange(split,size),x_test)
    # plt.show()
    ### Fourier
    f = fourier(num_freqs=num_freqs_fourier)
    if normalize:
        scaled = f.scale(x_train)
        if "omegas" in fit_kwargs.keys():
            freqs_input = [1/i for i in fit_kwargs["omegas"]]
            f.fit(scaled, freqs = freqs_input, iterations = 1000,verbose = True)
        f.fit(scaled, iterations = 1000,verbose = True)
        print(1/f.freqs)
        xhat_fourier = f.predict(size)
        xhat_fourier = f.descale(xhat_fourier)
    else:
        f.fit(x_train, iterations = 1000,verbose = True)
        print(1/f.freqs)
        xhat_fourier = f.predict(size)
    ### koopman
    model_object = fully_connected_mse(x_dim=x_train.shape[1], num_freqs=num_freqs_koopman, n_neurons=n_neurons,n_layers=n_layers)
    k = koopman(model_object, device='cpu',sample_num = sample_num)
    if normalize:
        k.fit(k.scale(x_train),**fit_kwargs)
        xhat_koopman = k.predict(size)
        xhat_koopman = k.descale(xhat_koopman)
    else:
        k.fit(x_train, **fit_kwargs)
        xhat_koopman = k.predict(size)
    ### plot
    #no zoom
    fig,axs = plot(x_train,x_test,xhat_fourier,xhat_koopman,split,size)
    plt.tight_layout()
    plt.savefig(f"figures/{figname}.png")
    plt.show()
    #zoom
    fig,axs = plot(x_train,x_test,xhat_fourier,xhat_koopman,split,size)
    xlim = (split-zoom[0], split+zoom[1])
    for ax in axs.flat:
        ax.set(xlim=xlim)
    plt.tight_layout()
    plt.savefig(f"figures/{figname}_zoom.png")
    plt.show()


params_base = {
    "data": data.artificial,
    "data_kwargs": {
        "split_ratio": 0.5,
        "noise": False,
        "nonlinear": False,
        "size": 10000,
        },
    "normalize": False,
    "num_freqs_fourier": 2,
    "num_freqs_koopman": 2,
    "n_neurons": 24,
    "n_layers": 1,
    "fit_kwargs": {
        "lr_omega": 1e-3,
        "lr_theta": 1e-4,
        "cutoff": 100,
        "iterations": 500,
        "interval": 10,
        "verbose": True,
        },
    "figname": "artificial",
    "sample_num": 12,
    "zoom": (100,100),
    }



if __name__ == '__main__':
    params = params_base.copy()
    ### experiment 1
    main(**params)
    ### experiment 2
    params["data_kwargs"]["noise"] = True
    params["figname"] = "artificial_noise"
    main(**params)
    ### experiment 3
    params["data_kwargs"]["noise"] = False
    params["data_kwargs"]["nonlinear"] = True
    params["figname"] = "artificial_nonlinear"
    main(**params)
    ### experiment 4
    params["data_kwargs"]["noise"] = True
    params["figname"] = "artificial_noise_nonlinear"
    main(**params)
    ### experiment 5
    params["data"] = data.lorenz
    params["data_kwargs"] = {"split_ratio": 0.6}
    params["figname"] = "lorenz"
    params["num_freqs_fourier"] = 24
    params["num_freqs_koopman"] = 24
    params["n_neurons"] = 512
    params["n_layers"] = 3
    params["fit_kwargs"]["interval"] = 25
    params["fit_kwargs"]["iterations"] = 1000
    params["fit_kwargs"]["cutoff"] = 500
    params["fit_kwargs"]["lr_omega"] = 1e-5
    params["zoom"] = (1000,1000)
    params["normalize"] = True

    main(**params)
    ### experiment 6
    params["data"] = data.energy_consumption
    params["data_kwargs"] = {"split_ratio": 0.6}
    params["figname"] = "energy_consumption"
    params["num_freqs_fourier"] = 24
    params["num_freqs_koopman"] = 24
    params["n_neurons"] = 512
    params["n_layers"] = 2
    params["fit_kwargs"]["interval"] = 20
    params["fit_kwargs"]["iterations"] = 250
    params["fit_kwargs"]["cutoff"] = 150
    params["fit_kwargs"]["omegas"] = [24,168,8760]
    params["zoom"] = (300,600)
    params["sample_num"] = 24
    params["normalize"] = True
    main(**params)



