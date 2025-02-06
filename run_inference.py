from tasks.math_solver import MathSolver

if __name__ == "__main__":
    # 示例：解方程 x² + 2x - 3 = 0，正确答案为 1 和 -3
    solver = MathSolver(
        equation="x² + 2x - 3 = 0",
        ground_truth="1,-3"
    )
    result_tree = solver.run(iterations=5)
    print("\n最终树结构:")
    solver.tree.print_tree(max_depth=5)