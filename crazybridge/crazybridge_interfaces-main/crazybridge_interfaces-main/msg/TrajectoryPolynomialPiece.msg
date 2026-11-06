# One piece of a piecewise polynomial trajectory (same layout as crazyswarm2).
# Each poly_* holds 8 coefficients in ascending powers (c0 + c1 t + ... + c7 t^7),
# evaluated in piece-local time t in [0, duration].
float32[] poly_x
float32[] poly_y
float32[] poly_z
float32[] poly_yaw
builtin_interfaces/Duration duration
