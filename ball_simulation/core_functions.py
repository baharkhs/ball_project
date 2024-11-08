class CoreFunctions:
    def __init__(self, position, velocity, acceleration):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

    def update_position(self, time_step):
        # Update the position based on velocity and acceleration
        self.position += self.velocity * time_step + 0.5 * self.acceleration * time_step ** 2
        self.velocity += self.acceleration * time_step
        return self.position
