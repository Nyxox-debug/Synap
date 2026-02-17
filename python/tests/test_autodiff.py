# NOTE: This has been removed
from synap import Value

def banner(msg):
    print(f"\n=== {msg} ===")

banner("scalar autodiff")

x = Value(2.0)
y = Value(3.0)

z = x * y + x
z.backward()

print("x.grad =", x.grad)
print("y.grad =", y.grad)

assert abs(x.grad - 4.0) < 1e-6
assert abs(y.grad - 2.0) < 1e-6

print("\nScalar autodiff test passed ✔️")
