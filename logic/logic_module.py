import numpy as np

try:
    from config import LOGIC_CONFIG
except ImportError:
    LOGIC_CONFIG = {
        'lookahead_distance': 0.5,  # meters
        'wheelbase': 0.3, # meters
        'max_cone_pairs': 4,
    }

class LogicModule:
    def __init__(self):
        self.l = LOGIC_CONFIG['lookahead_distance']
        self.WD = LOGIC_CONFIG['wheelbase']
        self.max_p = LOGIC_CONFIG['max_cone_pairs']


    def cone_sorting(self, world_cones):
        blue = [c for c in world_cones if c[2] == "Blue"]
        yellow = [c for c in world_cones if c[2] == "Yellow"]

        blue.sort(key=lambda c: c[1])   # sort by distance (z)
        yellow.sort(key=lambda c: c[1])
        return blue, yellow

    def cone_midpoints(self, blue_cones, yellow_cones, img=None):
        midpoints = []
        n = min(len(blue_cones), len(yellow_cones), self.max_p)

        for i in range(n):
            bx, bz, _, bu, bv = blue_cones[i]
            yx, yz, _, yu, yv = yellow_cones[i]

            x = (bx + yx) / 2.0
            z = (bz + yz) / 2.0
            u = int((bu + yu) / 2.0)
            v = int((bv + yv) / 2.0)

            midpoints.append((x, z, u, v))




        if len(yellow_cones) == 0 and len(blue_cones) >= 2: #right turn
            #for i in len(blue_cones):
            width = 60
            x0, z0 = blue_cones[0][0], blue_cones[0][1]
            x1, z1 = blue_cones[1][0], blue_cones[1][1]

            dx = x1 - x0
            dz = z1 - z0

            halfway_x = x0 + dx / 2.0
            halfway_z = z0 + dz / 2.0

            length = np.hypot(dx, dz)
            if length > 1e-6:
                dir_x = dx / length
                dir_z = dz / length

                perp_x = dir_x
                perp_z = -dir_z

                midpoint_x = halfway_x + perp_x * width / 2.0
                midpoint_z = halfway_z + perp_z * width / 2.0

                midpoints.append((midpoint_x, midpoint_z))

        if len(blue_cones) == 0 and len(yellow_cones) >= 2: #left turn
            # for i in len(yellow_cones):
            width = 60
            x0, z0 = yellow_cones[0][0], yellow_cones[0][1]
            x1, z1 = yellow_cones[1][0], yellow_cones[1][1]

            dx = x1 - x0
            dz = z1 - z0

            halfway_x = x0 + dx / 2.0
            halfway_z = z0 + dz / 2.0

            length = np.hypot(dx, dz)
            if length > 1e-6:
                dir_x = dx / length
                dir_z = dz / length

                perp_x = dir_x
                perp_z = -dir_z

                midpoint_x = halfway_x + perp_x * width / 2.0
                midpoint_z = halfway_z + perp_z * width / 2.0

                midpoints.append((midpoint_x, midpoint_z))

        return midpoints



    def Interpolation(self, midpoints):
        """
        Accepts midpoints as list of (x, z, u, v).
        Returns target (x_t, z_t) in local coordinates or None.
        """
        if not midpoints:
            return None


        #mulig fejl i kode!!
        midpoints.sort(key=lambda m: m[1])
        xs = [m[0] for m in midpoints]
        zs = [m[1] + self.WD for m in midpoints] # apply width offset to z-values

        x0, z0 = xs[0], zs[0]
        dist0 = np.hypot(x0, z0)

        if dist0 >= self.l:
            s = self.l / dist0
            return (s * x0, s * z0)

        # need at least two points for quadratic solve
        if len(xs) < 2:
            return None
        x1, z1 = xs[1], zs[1]

        a = x1 - x0
        b = z1 - z0

        A = a**2 + b**2
        B = 2*(x0 * a + z0 * b)
        C = x0**2 + z0**2 - self.l**2
        D = B**2 - 4*A*C

        if D < 0:
            return None

        r1 = (-B + np.sqrt(D)) / (2*A)
        r2 = (-B - np.sqrt(D)) / (2*A)
        roots = [r for r in (r1, r2) if r > 0]
        if not roots:
            return None

        s = roots[0]
        return (x0 + s*x1, z0 + s*z1)

    def steering_angle(self, target):
        """
        Calculate steering angle using pure pursuit algorithm.
        Returns steering angle in degrees.
        """
        if target is None:
            return None

        x, z = target

        # Avoid division by zero
        if abs(x) < 1e-6:
            return 0.0

        # Calculate turning radius
        r = self.l ** 2 / (2.0 * x)

        # Calculate steering angle
        steering_rad = np.arctan(self.WD / r)
        return float(np.degrees(steering_rad))