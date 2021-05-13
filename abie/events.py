class ParticleException(Exception):
    """
    Base class for the particle exception events.
    """

    def __init__(self, t, obj1, obj2, distance):
        self.t = t
        self.obj1 = obj1
        self.obj2 = obj2
        self.distance = distance


class CloseEncounterException(ParticleException):
    """
    Exception for the close encounter events between two particles.
    """

    def __str__(self):
        return (
            "Close encounter detected between particle #%d and particle #%d with a distance of %g at t = %g"
            % (self.obj1, self.obj2, self.distance, self.t)
        )


class CollisionException(ParticleException):
    """
    Exception for the collision events between two particles.
    """

    def __str__(self):
        return (
            "Collision detected between particle #%d and particle #%d with a distance of %g at t = %g"
            % (self.obj1, self.obj2, self.distance, self.t)
        )
