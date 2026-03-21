from vpython import *

# 1. Canvas Setup - Centered on the SJJ
scene = canvas(title='Superjumbojumbo Mega-Hub', width=1000, height=700, 
               center=vector(0,0,0), background=vector(0.4, 0.6, 1.0))

# --- Colors ---
grass_color = vector(0.1, 0.5, 0.1)
runway_color = vector(0.2, 0.2, 0.2)

# 2. Ground and Runway (The Stage)
ground = box(pos=vector(0, -16, 0), size=vector(1000, 1, 1000), color=grass_color)
# Main Runway directly under the plane
main_runway = box(pos=vector(0, -15.5, 0), size=vector(800, 0.2, 120), color=runway_color)

# 3. THE SUPERJUMBOJUMBO (Center of the Universe)
sjj_group = compound([
    # Fuselage
    cylinder(pos=vector(-47.5,0,0), axis=vector(95,0,0), radius=15, color=color.white),
    sphere(pos=vector(47.5,0,0), radius=15, color=color.white),
    # Wings
    box(pos=vector(0, 2, 45), size=vector(45, 1.5, 80), axis=vector(1, 0, 0.1), color=color.white),
    box(pos=vector(0, 2, -45), size=vector(45, 1.5, 80), axis=vector(1, 0, -0.1), color=color.white),
    # Tail
    box(pos=vector(-40, 25, 0), size=vector(20, 30, 1), axis=vector(1,0.5,0), color=color.white)
])

# 4. The 6 Engines (Adding them separately so they look right)
for z_off in [30, 50, 70, -30, -50, -70]:
    cylinder(pos=vector(-10, -8, z_off), axis=vector(15,0,0), radius=5, color=vector(0.3, 0.3, 0.3))

# 5. SCALE COMPARISON: The A380 (Cyan) - Placed nearby
a380_pos = vector(0, -10, 150) # To the side
a380_body = cylinder(pos=a380_pos + vector(-35,0,0), axis=vector(70,0,0), radius=7, color=color.cyan)
a380_wings = box(pos=a380_pos + vector(0,0,0), size=vector(30, 1, 80), color=color.cyan)

# 6. SCALE COMPARISON: The 737 (Red) - Placed nearby
b737_pos = vector(0, -13, -150) # To the other side
b737_body = cylinder(pos=b737_pos + vector(-15,0,0), axis=vector(30,0,0), radius=3, color=color.red)
b737_wings = box(pos=b737_pos + vector(0,0,0), size=vector(10, 0.5, 35), color=color.red)

# 7. Terminal
terminal = box(pos=vector(150, 0, 250), size=vector(100, 50, 50), color=color.gray(0.7))

print("Look at the center of the screen! The Superjumbojumbo is the white giant.")