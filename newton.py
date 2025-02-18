import math

#----------------------------------------------------- Main Logic -------------------------------------------------------
def main() -> None:

    G = 6.67428e-11     # Gravitational constant in [m^3*kg^-1*s^-1] 
    M = 5.972e24        # Earth mass in [kg]
    R = 6378137         # Radius earth in [m] 
    h = 420000          # ISS orbit height in [m]

    # v^2 = G*M/(R+h) velocity (v)
    speed = math.sqrt(G*M/(R+h))
    print(f"The calculated ISS speed is {speed/1000} in kmps")

    # T = 2*pi*(R+h)/v
    period = 2*math.pi*(R+5)/(speed)

    # 60 seoncds = 1 minute
    print(f"The calculated ISS orbit period is {period/60:.2f} in minutes")

if __name__ == "__main__":
    main()

 #----------------------------------------------------- Main Logic -------------------------------------------------------   