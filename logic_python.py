import numpy as np
class ImaginaryNumber:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        """
        w = Scalar (Real) part
        x, y, z = Vector (Imaginary) parts
        Defaults to the identity quaternion (no rotation).
        """
        self.w = np.float32(w)
        self.x = np.float32(x)
        self.y = np.float32(y)
        self.z = np.float32(z)

    def __mul__(self, other):
        """Standard Hamilton product for quaternions."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return ImaginaryNumber(w, x, y, z)

    def __rmul__(self, other):
        return self.__mul__(other)

    def norm(self):
        """Returns the magnitude of the quaternion."""
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        """Returns a normalized unit quaternion (required for rotations)."""
        n = self.norm()
        if n == 0:
            return ImaginaryNumber(1.0, 0.0, 0.0, 0.0)
        return ImaginaryNumber(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self):
        """Returns the conjugate (negates the vector part)."""
        return ImaginaryNumber(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        """Returns the inverse of the quaternion."""
        n_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        conj = self.conjugate()
        return ImaginaryNumber(conj.w / n_sq, conj.x / n_sq, conj.y / n_sq, conj.z / n_sq)

    def rotate_vector(self, v):
        """
        Rotates a 3D vector (numpy array or list) by this quaternion.
        v_prime = q * v * q^-1
        """
        # Convert vector to a pure quaternion (w=0)
        v_quat = ImaginaryNumber(0.0, v[0], v[1], v[2])
        
        # Multiply: q * v * q_inv
        # (Assuming self is a unit quaternion, inverse == conjugate)
        q_inv = self.conjugate()
        res_quat = self * v_quat * q_inv
        
        return np.array([res_quat.x, res_quat.y, res_quat.z], dtype=np.float32)

    @staticmethod
    def from_axis_angle(axis, angle):
        """
        Creates a rotation quaternion from a 3D axis and an angle (in radians).
        Axis must be a normalized 3D numpy array.
        """
        axis = np.array(axis, dtype=np.float32)
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
            
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        
        return ImaginaryNumber(
            np.cos(half_angle),
            axis[0] * s,
            axis[1] * s,
            axis[2] * s
        )

    def __repr__(self):
        return f"Quaternion({self.w:.4g}, {self.x:.4g}i, {self.y:.4g}j, {self.z:.4g}k)"
    
class Biquaternion:
    def __init__(self, w=1.0+0j, x=0j, y=0j, z=0j):
        """
        w, x, y, z are Python complex numbers.
        """
        self.w = complex(w)
        self.x = complex(x)
        self.y = complex(y)
        self.z = complex(z)

    def __mul__(self, other):
        """Standard Hamilton product works seamlessly with complex coefficients."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Biquaternion(w, x, y, z)

    def quat_conjugate(self):
        """Standard quaternion conjugate."""
        return Biquaternion(self.w, -self.x, -self.y, -self.z)

    def complex_conjugate(self):
        """Conjugates the Python complex coefficients."""
        return Biquaternion(np.conj(self.w), np.conj(self.x), np.conj(self.y), np.conj(self.z))

    def hermitian_conjugate(self):
        """Required for 4D spacetime transformations (Q^dagger)."""
        return self.complex_conjugate().quat_conjugate()

    def normalize(self):
        """Normalizes the biquaternion."""
        # For Lorentz transformations, Q * Q_complex_conjugate should equal 1
        norm_sq = self.w**2 + self.x**2 + self.y**2 + self.z**2
        norm = np.sqrt(norm_sq)
        if norm == 0:
            return Biquaternion(1.0+0j, 0j, 0j, 0j)
        return Biquaternion(self.w/norm, self.x/norm, self.y/norm, self.z/norm)

    def apply_to_4vector(self, v_4d):
        """
        Applies a Lorentz transformation (Rotation + Boost) to a 4D numpy array.
        v_4d = [t, x, y, z]
        """
        # 1. Map 4-vector to Biquaternion: V = t + I*(x*i + y*j + z*k)
        V = Biquaternion(
            w = v_4d[0], 
            x = v_4d[1] * 1j, 
            y = v_4d[2] * 1j, 
            z = v_4d[3] * 1j
        )
        
        # 2. Apply transformation: V' = Q * V * Q^dagger
        Q_dag = self.hermitian_conjugate()
        V_prime = self * V * Q_dag

        # 3. Map back to 4D numpy array
        # Time stays real, Space was multiplied by 1j, so we extract the imaginary parts
        return np.array([
            V_prime.w.real,
            V_prime.x.imag,
            V_prime.y.imag,
            V_prime.z.imag
        ], dtype=np.float32)

    @staticmethod
    def from_axis_angle(axis, angle):
        """Creates a pure spatial rotation SO(3)."""
        axis = np.array(axis, dtype=np.float32)
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
            
        half_angle = angle / 2.0
        s = np.sin(half_angle)
        
        # Note: Your padded logic uses axis[1], axis[2], axis[3] 
        return Biquaternion(np.cos(half_angle), axis[1]*s, axis[2]*s, axis[3]*s)

    @staticmethod
    def from_boost(direction, velocity, c_speed=1.0):
        """
        Creates a Lorentz boost along a 3D direction vector.
        """
        direction = np.array(direction, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            
        beta = velocity / c_speed
        if beta >= 0.9999: beta = 0.9999 # Enforce universal speed limit
        
        # Rapidity phi: tanh(phi) = beta
        phi = np.arctanh(beta)
        half_phi = phi / 2.0
        
        ch = np.cosh(half_phi)
        sh = np.sinh(half_phi)
        
        # Boost Biquaternion uses imaginary angles!
        return Biquaternion(
            w = ch,
            x = direction[1] * sh * 1j,
            y = direction[2] * sh * 1j,
            z = direction[3] * sh * 1j
        )