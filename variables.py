class param:

    scale = 1
    T = 150
    t_back = -20
    t_fore = 20

    pixel_x = 28
    m = pixel_x*pixel_x #Number of neurons in first layer
    n =  2 #Number of neurons in second layer
    Pref = 0.
    Prest = 0.
    Pmin = -5.0
    Pth = 80
    D = 0.75

    w_max = 2.0
    w_min = -1.2
    sigma = 0.02 
    A_plus = 0.8  
    A_minus = 0.3 
    tau_plus = 10
    tau_minus = 10

    epoch = 35


    fr_bits = 12
    int_bits = 12