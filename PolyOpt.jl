
module PolyOpt
export scaling_poly, polynomial_minimization


using DynamicPolynomials
using TSSOS
using HomotopyContinuation

"""
Estimate the scaling factor for the variables of a polynomial to make TSSOS computations numerically stable

    see https://github.com/wangjie212/TSSOS/issues/6
"""
function scaling_poly(p::Polynomial)
    X = transpose(hcat([exponents(t) for t in terms(p)]...))

    # Get the scaling via linear regression
    scaling = X \ log.(abs.(coefficients(p)))

    exp.(abs.(scaling))
end


"""
Try TSSOS, scaled TSSOS, and Homotopy Continuation to get the global minima of the polynomial
"""
function polynomial_minimization_TEST(p::Polynomial)
    ################################################################################################
    #
    #   Try HomotopyContinuation
    #
    ################################################################################################
    println("\n*********** Try HomotopyContinuation ***********\n")

    # Find the critical points
    result = solve(differentiate.(p, variables(p)))
    critical_points = real_solutions(result)

    # Get the exact values for the exact objective function for the found critical points
    val_p = p.(critical_points)

    minimizer_homotopy = critical_points[argmin(val_p)]

    optimum = minimum(val_p)
    @show optimum

    @show minimizer_homotopy

    ################################################################################################
    #
    #   Try just plain TSSOS
    #
    ################################################################################################
    println("\n*********** Try just plain TSSOS ***********\n")

    opt,sol,data = tssos_first(p, variables(p), QUIET=true, solution=true, newton=false);
    previous_sol = sol

    while ~isnothing(sol)
        previous_sol = sol
        opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
    end

    minimizer_tssos = previous_sol

    @show minimizer_tssos
    ################################################################################################
    #
    #   Try TSSOS on polynomial with scaled variables
    #
    ################################################################################################
    println("\n*********** Try TSSOS on polynomial with scaled variables ***********\n")

    # find variable scaling
    scale = scaling_poly(p)

    # scale the polynomial
    p_scaled = subs(p, variables(p) => scale .* variables(p))

    # minimize
    opt,sol,data = tssos_first(p_scaled, variables(p), QUIET=true, solution=true, newton=false);
    previous_sol = sol

    while ~isnothing(sol)
        previous_sol = sol
        opt,sol,data = tssos_higher!(data; QUIET=true, solution=true);
    end

    minimizer_scaled_tssos = scale .* previous_sol

    @show minimizer_scaled_tssos
    ################################################################################################
    #
    #   Comparing
    #
    ################################################################################################

    minimizers_found = [[minimizer_homotopy] [minimizer_tssos], [minimizer_scaled_tssos]]
    val_p = p.(minimizers_found)

    minimizers_found[argmin(val_p)]
end

end;