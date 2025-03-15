using DataFrames
using JuMP, GLPK

solutions = Vector{Vector{Int}}()
c = Vector{Float64}()

total_length = 7.4

for x in 0:floor(Int, total_length / 2.9)
    for y in 0:floor(Int, total_length / 2.1)
        for z in 0:floor(Int, total_length / 1.5)
            if x == 0 && y == 0 && z == 0
                continue
            end

            object = 2.9 * x + 2.1 * y + 1.5 * z
            if object <= total_length
                push!(solutions, [x, y, z])
                push!(c, total_length - object)
            end
        end
    end
end

A = zeros(3, length(solutions))

for i in eachindex(solutions)
    A[:, i] = solutions[i]
end

model = Model(GLPK.Optimizer)

@variable(model, x[1:length(c)] >= 0, Int)

@objective(model, Min, c' * x)

@constraint(model, A * x .== 100)
@constraint(model, x .>= 0)

optimize!(model)

df = hcat(DataFrame(Int.(A)', ["2.9m", "2.1m", "1.5m"]), DataFrame("Spend" => c, "Remain" => A' * [2.9, 2.1, 1.5], "Count" => Int.(value.(x))))
println(df)

println("Total: ", Int(sum(value.(x))))
println("Min waste:", round(objective_value(model), digits=1))
