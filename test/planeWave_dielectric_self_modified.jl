using SphericalScattering
using Test
using BEAST
using LinearAlgebra
using StaticArrays
    

ω = (2*pi)*1e7
f= ω/(2*pi)

𝜇 = SphericalScattering.μ0
𝜀 = SphericalScattering.ε0
#𝜇 = 1.0
#𝜀 = 1.0

# Embedding
μ2 =𝜇 * 1.0
ε2 =𝜀 * 3.0

# Filling
μ1 = 𝜇 * 1.0
ε1 = 𝜀 * (2.0 + 3im)


c2 = 1 / sqrt(ε2 * μ2)
c1 = 1 / sqrt(ε1 * μ1)

k2 = 2π * f / c2
k1 = 2π * f / c1

η2 = sqrt(μ2 / ε2)
η1 = sqrt(μ1 / ε1)



# BASIS AUF DEM RAND DER KUGEL
spRadius=1.0
meshparam=0.2
sphere = BEAST.meshsphere(spRadius, meshparam) # das ist jetzt nur die Kugelfläche!!!!!!

RT=raviartthomas(sphere)

points_cartNF = Vector([SVector(-2.0,0.0, 0.0),SVector(2.0,0.0, 0.0)]) #außerhalb wenn radius 1

points_cartNF_inside = Vector([SVector(0.2, 0.0, 0.1),SVector(-0.1, 0.3, 0.05)]) # Ursprung geht nicht...

points_cartFF = Vector([SVector(1.0, 0.0, 0.0),SVector(1.0, 0.0, 0.0)])






# MoM solution via PMCHWT
𝓣k2 = Maxwell3D.singlelayer(; wavenumber=k2, alpha=-im * μ2 * ω, beta=1 / (-im * ε2 * ω))
𝓣k1 = Maxwell3D.singlelayer(; wavenumber=k1, alpha=-im * μ1 * ω, beta=1 / (-im * ε1 * ω))

𝓚k2 = Maxwell3D.doublelayer(; wavenumber=k2)
𝓚k1 = Maxwell3D.doublelayer(; wavenumber=k1)

𝐸 = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=k2)
 
𝒆 = (n × 𝐸) × n
H = (-1 / (im * μ2 * ω)) * curl(𝐸)
𝒉 = (n × H) × n
nx𝒉 = (n × H)

Tk2 = Matrix(assemble(𝓣k2, RT, RT))
Tk1 = Matrix(assemble(𝓣k1, RT, RT))

Kk2_rt = Matrix(assemble(𝓚k2, RT, RT))
Kk1_rt = Matrix(assemble(𝓚k1, RT, RT))

e = Vector(assemble(𝒆, RT))
    h = Vector(assemble(𝒉, RT))

Z_PMCHWT = [
    -(Kk2_rt + Kk1_rt)  (Tk2 + Tk1)./η2
    (((1 / η2)^2 .* Tk2 + (1 / η1)^2 .* Tk1).*η2)  (Kk2_rt+Kk1_rt)
]

eh = [-e; -h .* η2]

mj_PMCHWT = Z_PMCHWT \ eh

m = mj_PMCHWT[1:numfunctions(RT)]
j = mj_PMCHWT[(1 + numfunctions(RT)):end] ./ η2

function efield(𝓣, j, X_j, 𝓚, m, X_m, pts)
    return potential(MWSingleLayerField3D(𝓣), pts, j, X_j) .+ potential(BEAST.MWDoubleLayerField3D(𝓚), pts, m, X_m)
end

function hfield(𝓣, m, X_m, 𝓚, j, X_j, pts)
    return potential(MWSingleLayerField3D(𝓣), pts, m, X_m) .+ potential(BEAST.MWDoubleLayerField3D(𝓚), pts, j, X_j)
end

function efarfield(𝓣, j, X_j, 𝓚, m, X_m, pts)
    return potential(MWFarField3D(𝓣), pts, j, X_j) .+ potential(BEAST.MWDoubleLayerFarField3D(𝓚), pts, m, X_m)
end

EF₂MoM = efield(𝓣k2, j, RT, 𝓚k2, -m, RT, points_cartNF)
EF₁MoM = efield(𝓣k1, -j, RT, 𝓚k1, +m, RT, points_cartNF_inside)

sp = DielectricSphere(; radius=spRadius, embedding=Medium(ε2, μ2), filling=Medium(ε1, μ1))
ex = planeWave(sp; frequency=f)



# E-Field
EF₂ = scatteredfield(sp, ex, ElectricField(points_cartNF))
EF₁ = scatteredfield(sp, ex, ElectricField(points_cartNF_inside))

diff_EF₂ = norm.(EF₂ - EF₂MoM) ./ maximum(norm.(EF₂))  # worst case error
diff_EF₁ = norm.(EF₁ - EF₁MoM) ./ maximum(norm.(EF₁))  # worst case error

#@test maximum(20 * log10.(abs.(diff_EF₂))) < -25 # dB 
#@test maximum(20 * log10.(abs.(diff_EF₁))) < -25 # dB 

@show maximum(abs.(diff_EF₂))
@show maximum(abs.(diff_EF₁))



# H-Field
HF₂MoM = hfield(𝓣k2, +(1 / η2)^2 .* m, RT, 𝓚k2, +j, RT, points_cartNF)
HF₁MoM = hfield(𝓣k1, -(1 / η1)^2 .* m, RT, 𝓚k1, -j, RT, points_cartNF_inside)

HF₂ = scatteredfield(sp, ex, MagneticField(points_cartNF))
HF₁ = scatteredfield(sp, ex, MagneticField(points_cartNF_inside))

diff_HF₂ = norm.(HF₂ - HF₂MoM) ./ maximum(norm.(HF₂))  # worst case error
diff_HF₁ = norm.(HF₁ - HF₁MoM) ./ maximum(norm.(HF₁))  # worst case error

#@test maximum(20 * log10.(abs.(diff_HF₂))) < -24 # dB 
#@test maximum(20 * log10.(abs.(diff_HF₁))) < -25 # dB 

@show maximum(abs.(diff_HF₂))
@show maximum(abs.(diff_HF₁))



# Far-Field
FF_MoM = -im * f / (2 * c2) * efarfield(𝓣k2, j, RT, 𝓚k2, -m, RT, points_cartFF)
FF = scatteredfield(sp, ex, FarField(points_cartFF))

diff_FF = norm.(FF - FF_MoM) ./ maximum(norm.(FF))  # worst case error
@show maximum(abs.(diff_FF))
#@test maximum(20 * log10.(abs.(diff_FF))) < -24 # dB

