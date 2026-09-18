#include <polyfem/State.hpp>
#include <polyfem/varforms/VarForm.hpp>
#include <polyfem/utils/Logger.hpp>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace polyfem;

int main(int argc, char **argv)
{
	const std::string scene = (argc > 1)
		? argv[1]
		: "/Users/sak9889/WORK/GitHub/Polyfem/polyfem-data/contact/examples/"
	"2D/golf-ball-doformable-wall.json";

	const std::string out_dir = (argc > 2) ? argv[2] : "/tmp/scratch-out";
	std::filesystem::create_directories(out_dir);

	json in_args;
	{
		std::ifstream f(scene);
		if (!f.is_open())
		{
			std::cerr << "cannot open " << scene << "\n";
			return EXIT_FAILURE;
		}
		f >> in_args;
	}

	// lets relative mesh paths and the "common" include resolve
	in_args["root_path"] = scene;
	in_args["/output/directory"_json_pointer] = std::filesystem::absolute(out_dir).string();
	in_args["/output/log/level"_json_pointer] = 1; // debug

	State state;
	state.init(in_args, /*strict_validation=*/true);
	state.load_mesh(/*non_conforming=*/true);

	Eigen::MatrixXd sol;
	state.solve(sol);

	logger().info("sol: {} x {}", sol.rows(), sol.cols());
	logger().info("max |u| = {}", sol.cwiseAbs().maxCoeff());

	state.variational_formulation->save_json(sol);
	state.variational_formulation->export_data(sol);

	return EXIT_SUCCESS;
}