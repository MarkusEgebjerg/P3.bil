import numpy as np
import cv2

class LogicModule:
    def __init__(self, lookahead=0.5, track_width=0.3, max_pairs=4):
        self.l = lookahead
        self.WD = track_width
        self.max_p = max_pairs

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


        for i in len(blue_cones):
            if blue_cones[i][0] >= 1280/2: right_turn = 1

        for i in len(yellow_cones):
            if yellow_cones[i][0] <= 1280/2: left_turn = 1


        if len(yellow_cones) == 0 and right_turn == 1: #right turn
            #for i in len(blue_cones):
            width = 80
            first_vector = (blue_cones[0][0], blue_cones[0][1]) #vector/point to first blue cone
            second_vector = (blue_cones[1][0] - blue_cones[0][0], blue_cones[1][1] - blue_cones[0][1]) #vector from first blue cone to second blue cone
            halfway_point = (first_vector[0] - second_vector[0]/2  , first_vector[1] - second_vector[1]/2) #Halfwaypoint between first blue cone and second blue cone
            length = np.hypot(second_vector[0], second_vector[1])

            second_vector_para = (second_vector[0]/length, second_vector[1]/length)
            third_vector = (second_vector_para[0] * width/2 * 1 , second_vector_para[1] * width/2 * -1)
            midpoint = (third_vector[0] + halfway_point[0], third_vector[1] + halfway_point[1])

            midpoints.append(midpoint)

        if len(blue_cones) == 0 and left_turn == 1: #left turn

            width = 80
            first_vector = (blue_cones[0][0], blue_cones[0][1])  # vector/point to first blue cone
            second_vector = (blue_cones[1][0] - blue_cones[0][0],
                             blue_cones[1][1] - blue_cones[0][1])  # vector from first blue cone to second blue cone
            halfway_point = (first_vector[0] - second_vector[0] / 2, first_vector[1] - second_vector[
                1] / 2)  # Halfwaypoint between first blue cone and second blue cone
            length = np.hypot(second_vector[0], second_vector[1])

            second_vector_para = (second_vector[0] / length, second_vector[1] / length)
            third_vector = (second_vector_para[0] * width / 2 * 1, second_vector_para[1] * width / 2 * -1)
            midpoint = (third_vector[0] + halfway_point[0], third_vector[1] + halfway_point[1])

            midpoints.append(midpoint)



            # Draw midpoints if image is provided
            if img is not None:
                try:
                    cv2.circle(img, (u, v), 4, (0, 255, 0), -1)
                    cv2.putText(img, f"{round(x,2)},{round(z,2)}", (u+5, v-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                except Exception:
                    pass

        return midpoints



    def Interpolation(self, midpoints):
        """
        Accepts midpoints as list of (x, z, u, v).
        Returns target (x_t, z_t) in local coordinates or None.
        """
        if not midpoints:
            return None


        #mulig fejl i kode!!
        xs = sorted([m[0] for m in midpoints])
        zs = sorted([m[1] + self.WD for m in midpoints])  # apply width offset to z-values

        #ændring: Sorter hver tuple i midpoints før de bliver adskilt. Så vi sørger for at fx x_0 og Z_0 passer sammen for middpunktet.
        #midpoints.sort(key=lambda x: x[1])
        #xs = [m[0] for m in midpoints]
        #zs = [m[1] + self.WD for m in midpoints] # apply width offset to z-values

        x0, z0 = xs[0], zs[0]
        dist0 = np.hypot(x0, z0)

        if dist0 >= self.l:
            s = self.l / dist0
            return (s * x0, s * z0)

        # need at least two points for quadratic solve
        if len(xs) < 2:
            return None
        x1, z1 = xs[1], zs[1]

        a = x1 - x0  # defined for easier following computation
        b = z1 - z0  # defined for easier following computation

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