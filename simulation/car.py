import math
import numpy as np


class Car:
    def __init__(
            self, 
            mass, 
            position, 
            heading = np.array([1.0, 0.0]), # unit vector
            steering_range = 45, # degrees tires can turn from center 
            acceleration_rate = 0.4,
            braking_rate = 0.4,
        ):
        self.mass = mass
        self.position = position
        self.heading = heading
        self.speed = 0.0
        self.steering_range = steering_range
        self.acceleration_rate = acceleration_rate
        self.braking_rate = braking_rate
        self.rotation = math.atan2(heading[1], heading[0])
        self.rays = []
        self.time = 0.0

        self.size = (20, 10)  # width, length
        
    def update(self, throttle, brake, steering, delta_time):
        # Compute acceleration
        accel = self.acceleration_rate * throttle
        decel = self.braking_rate * brake
        net_accel = accel - decel
        self.time += delta_time

        # Compute velocity
        self.speed += net_accel * delta_time
        self.speed *= 0.99  # slows down over time
        for _ in range(5):
            forward = self.steer_vector(steering)

            offset = self.heading * self.size[0] * 0.5

            front = self.position + offset
            back = self.position - offset
            front += forward * self.speed * delta_time * 0.2
            direction = back - front
            self.heading = -direction / np.linalg.norm(direction)
            self.rotation = math.atan2(self.heading[1], self.heading[0]) * (180 / math.pi)
            self.position = front - self.heading * self.size[0] / 2

    def steer_vector(self, steering_input):
        return self.angle_to_vector(self.heading, -steering_input * self.steering_range)
    
    def angle_to_vector(self,vector, angle):
        x, y = vector
        # Convert steering input to rotation angle in radians
        angle_rad = math.radians(angle)
    
        # 2D rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
    
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
    
        return np.array([x_rot, y_rot], dtype=np.float32)
    
    def detect_obstacles(self, map):
        # Cast rays in three directions: left, forward,right
        far_left_ray = (self.position, self.angle_to_vector(self.heading, -90))
        mid_left_ray = (self.position, self.angle_to_vector(self.heading, -30))
        left_ray = (self.position, self.angle_to_vector(self.heading, -15))
        forward_ray = (self.position, self.heading)
        right_ray = (self.position, self.angle_to_vector(self.heading, 15))
        mid_right_ray = (self.position, self.angle_to_vector(self.heading, 30))
        far_right_ray = (self.position, self.angle_to_vector(self.heading, 90))

        far_left_dist = self.ray_cast(far_left_ray, map.rectangles)
        mid_left_dist = self.ray_cast(mid_left_ray, map.rectangles)
        left_dist = self.ray_cast(left_ray, map.rectangles)
        forward_dist = self.ray_cast(forward_ray, map.rectangles)
        right_dist = self.ray_cast(right_ray, map.rectangles)
        mid_right_dist = self.ray_cast(mid_right_ray, map.rectangles)
        far_right_dist = self.ray_cast(far_right_ray, map.rectangles)

        self.rays = [
            (*far_left_ray[0], far_left_ray[0][0] + far_left_ray[1][0] * far_left_dist, far_left_ray[0][1] + far_left_ray[1][1] * far_left_dist),
            (*mid_left_ray[0], mid_left_ray[0][0] + mid_left_ray[1][0] * mid_left_dist, mid_left_ray[0][1] + mid_left_ray[1][1] * mid_left_dist),
            (*left_ray[0], left_ray[0][0] + left_ray[1][0] * left_dist, left_ray[0][1] + left_ray[1][1] * left_dist),
            (*forward_ray[0], forward_ray[0][0] + forward_ray[1][0] * forward_dist, forward_ray[0][1] + forward_ray[1][1] * forward_dist),
            (*right_ray[0], right_ray[0][0] + right_ray[1][0] * right_dist, right_ray[0][1] + right_ray[1][1] * right_dist),
            (*mid_right_ray[0], mid_right_ray[0][0] + mid_right_ray[1][0] * mid_right_dist, mid_right_ray[0][1] + mid_right_ray[1][1] * mid_right_dist),
            (*far_right_ray[0], far_right_ray[0][0] + far_right_ray[1][0] * far_right_dist, far_right_ray[0][1] + far_right_ray[1][1] * far_right_dist),
        ]

        return far_left_dist, mid_left_dist, left_dist, forward_dist, right_dist, mid_right_dist, far_right_dist

    def ray_cast(self, ray, rectangles):
        origin, direction = ray
        origin = np.array(origin, dtype=np.float32)
        direction = np.array(direction, dtype=np.float32)

        min_distance = float('inf')

        for rect in rectangles:
            rx, ry, rw, rh, rot_deg = rect

            # To local space
            local_origin = origin - np.array([rx, ry], dtype=np.float32)

            theta = -math.radians(rot_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            rot_matrix = np.array([
                [cos_t, -sin_t],
                [sin_t,  cos_t]
            ], dtype=np.float32)

            local_origin = rot_matrix @ local_origin
            local_dir = rot_matrix @ direction

            min_bound = np.array([-rw/2, -rh/2], dtype=np.float32)
            max_bound = np.array([ rw/2,  rh/2], dtype=np.float32)

            tmin = -float('inf')
            tmax = float('inf')

            hit = True

            for i in range(2):
                if abs(local_dir[i]) < 1e-8:
                    # Ray parallel to slab
                    if local_origin[i] < min_bound[i] or local_origin[i] > max_bound[i]:
                        hit = False
                        break
                else:
                    t1 = (min_bound[i] - local_origin[i]) / local_dir[i]
                    t2 = (max_bound[i] - local_origin[i]) / local_dir[i]

                    t_near = min(t1, t2)
                    t_far = max(t1, t2)

                    tmin = max(tmin, t_near)
                    tmax = min(tmax, t_far)

                    if tmin > tmax:
                        hit = False
                        break

            if hit and tmax >= 0:
                t_hit = tmin if tmin >= 0 else tmax  # handle inside case

                if t_hit < min_distance:
                    min_distance = t_hit

        return min_distance

    def path_progress(self, map):
        progress, _ = self._closest_path_segment(map)
        return progress

    def path_tangent(self, map):
        _, tangent = self._closest_path_segment(map)
        return tangent

    def _closest_path_segment(self, map):
        closest_distance = float('inf')
        distance_along_path = 0.0  # from closest
        distance = 0.0  # total
        best_tangent = np.array([1.0, 0.0], dtype=np.float32)

        for line in map.path:
            x1, y1, x2, y2 = line
            line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            (dist_to_line, projected) = self.dist_to_line((x1, y1), (x2, y2))
            if dist_to_line < closest_distance and projected > -0.1:
                closest_distance = dist_to_line
                distance_along_path = distance + projected
                if line_length > 1e-8:
                    best_tangent = np.array(
                        [(x2 - x1) / line_length, (y2 - y1) / line_length],
                        dtype=np.float32,
                    )
            distance += line_length

        return distance_along_path / map.path_length, best_tangent
        
    def dist_to_line(self, line_start, line_end):
        p = np.array(self.position, dtype=np.float32)
        a = np.array(line_start, dtype=np.float32)
        b = np.array(line_end, dtype=np.float32)

        ab = b - a
        ap = p - a

        ab_len_sq = np.dot(ab, ab)

        if ab_len_sq == 0:
            distance = np.linalg.norm(p - a)
            return distance, 0.0

        # length on segment
        t = np.dot(ap, ab) / ab_len_sq
        t_clamped = np.clip(t, 0.0, 1.0)

        # Closest point on segment
        closest = a + t_clamped * ab

        # Distance to segment
        distance = np.linalg.norm(p - closest)

        projected_length = t_clamped * np.sqrt(ab_len_sq)

        return distance, projected_length