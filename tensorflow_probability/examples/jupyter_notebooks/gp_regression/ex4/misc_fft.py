
tf=300
dt=200
plt.figure(figsize=(12, 4))
plt.scatter(
            # np.mod(
                input_points_[:tf, 0]
                #    , dt)
                ,
            observations_[:tf],
            c='b',
            marker='o',
            label='Observations')
for i in range(num_predictive_samples):
    plt.scatter( 
                # np.mod(
        predictive_index_points_[:tf]
        # , dt)
        ,
            samples[i, :tf],
            c='r', alpha=.9,marker='.',
            label='Posterior Sample' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.grid()
plt.xlim(0,dt*1.2)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
# plt.savefig('./save/'+val+'_gpm_critic.png')
plt.show()

scipy.fft.fftfreq(observations_)


plt.plot(np.abs(scipy.fft(observations_))[:observations_.shape[0]//2] * 2.0 / observations_.shape[0]);plt.show()

np.sort(np.abs(scipy.fft(observations_))/ observations_.shape[0])

plt.plot(scipy.fft(observations_));plt.show()

plt.plot(scipy.fft(
    np.sin(np.arange(0,8*np.pi))
    ).real);plt.show()
"

nx=100
dx=0.0001
times0=np.linspace(0,nx*dx,nx)
freqs0=np.linspace(0,1.0/(2*dx),nx//2)

fcn = scipy.fft(np.sin(150*np.pi*times0) #+ np.sin(0.5*np.pi*times0))

print(fcn)
plt.plot(freqs0,
         np.abs(fcn[:nx//2]) * 2.0 / nx
         );plt.show()
