import taichi as ti
import taichi.math as tm

#ti.init(arch=ti.cpu) # Switch to ti.cuda or ti.metal for GPU acceleration

## Complex ##
@ti.func
def complex_mul(a, b):
    """Multiplies two complex numbers (represented as 2D vectors)."""
    # (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
    return ti.Vector([a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]])

@ti.func
def complex_div(a, b):
    """Divides two complex numbers (a / b)."""
    # (a * b_conjugate) / |b|^2
    b_conjugate = ti.Vector([b[0], -b[1]])
    mod_sq = b[0]**2 + b[1]**2
    return complex_mul(a, b_conjugate) / mod_sq

@ti.func
def complex_mod_sq(a):
    """Calculates the modulus squared |a|^2 of a complex number."""
    return a[0]**2 + a[1]**2

@ti.func
def complex_sqrt(z):
    """Complex square root: sqrt(z)"""
    mod_z = ti.sqrt(z[0]**2 + z[1]**2)
    real_part = ti.sqrt((mod_z + z[0]) / 2.0)
    imag_part = ti.math.sign(z[1]) * ti.sqrt((mod_z - z[0]) / 2.0)
    # Handle the case where the imaginary part is zero
    if z[1] == 0:
        imag_part = 0.0
    return ti.Vector([real_part, imag_part])

## Quaternions ###
ImaginaryNumber = ti.types.struct(real=ti.f64, imaginary=ti.f64, oneiric=ti.f64, numinous=ti.f64)
# Quaternion Full Operations 
@ti.func
def imaginary_add(q1, q2):
    return ImaginaryNumber(
        real=q1.real + q2.real, 
        imaginary=q1.imaginary + q2.imaginary, 
        oneiric=q1.oneiric + q2.oneiric, 
        numinous=q1.numinous + q2.numinous
    )

@ti.func
def imaginary_sub(q1, q2):
    return ImaginaryNumber(
        real=q1.real - q2.real, 
        imaginary=q1.imaginary - q2.imaginary, 
        oneiric=q1.oneiric - q2.oneiric, 
        numinous=q1.numinous - q2.numinous
    )

@ti.func
def imaginary_mul(q1, q2):
    """Cross-multiplying using standard Hamilton rules"""
    return ImaginaryNumber(
        real = q1.real*q2.real - q1.imaginary*q2.imaginary - q1.oneiric*q2.oneiric - q1.numinous*q2.numinous,
        imaginary = q1.real*q2.imaginary + q1.imaginary*q2.real + q1.oneiric*q2.numinous - q1.numinous*q2.oneiric,
        oneiric = q1.real*q2.oneiric - q1.imaginary*q2.numinous + q1.oneiric*q2.real + q1.numinous*q2.imaginary,
        numinous = q1.real*q2.numinous + q1.imaginary*q2.oneiric - q1.oneiric*q2.imaginary + q1.numinous*q2.real
    )

# Quaternion Partial Operations
@ti.func
def imaginary_mul_real(q, a):
    """Cross-multiplying using standard Hamilton rules"""
    return ImaginaryNumber(
        real = q.real*a,
        imaginary = q.imaginary*a,
        oneiric = q.oneiric*a,
        numinous = q.numinous*a
    )
@ti.func
def imaginary_add_real(q, a):
    """Cross-multiplying using standard Hamilton rules"""
    return ImaginaryNumber(
        real = q.real+a,
        imaginary = q.imaginary,
        oneiric = q.oneiric,
        numinous = q.numinous
    )
# Quaternion Coordinates
@ti.func
def imaginary_norm(q):
    """Calculates the Euclidean norm (magnitude) of the quaternion."""
    # Sum of squares of all components
    sum_sq = (q.real**2 + 
              q.imaginary**2 + 
              q.oneiric**2 + 
              q.numinous**2)
    return ti.sqrt(sum_sq)

@ti.func
def imaginary_norm_sqr(q):
    """Calculates the Euclidean norm (magnitude) of the quaternion."""
    # Sum of squares of all components
    sum_sq = (q.real**2 + 
              q.imaginary**2 + 
              q.oneiric**2 + 
              q.numinous**2)
    return sum_sq

@ti.func
def normalize_imaginary(q):
    return imaginary_mul_real(q, 1/(imaginary_norm(q)))

@ti.func
def imaginary_inverse(q):
    """q^-1 = q* / N(q)"""
    norm_sq = imaginary_norm_sqr(q)
    return ImaginaryNumber(
        real=q.real/norm_sq, 
        imaginary=-q.imaginary/norm_sq, 
        oneiric=-q.oneiric/norm_sq, 
        numinous=-q.numinous/norm_sq
    )

## Dual Numbers ##
