#include "AMIPSEnergy.hpp"

#include <polyfem/utils/Logger.hpp>

#include <polyfem/autogen/elastic_energies/AMIPS2d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS2drest.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3d.hpp>
#include <polyfem/autogen/elastic_energies/AMIPS3drest.hpp>


namespace polyfem::assembler
{
	void AMIPSEnergy::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);

		if (params.contains("use_rest_pose"))
		{
			use_rest_pose_ = params["use_rest_pose"].get<bool>();
		}

		if (energy_weights_.size() <= index)
			energy_weights_.resize(index + 1, 1.0);

		if (params.contains("weight"))
			energy_weights_[index] = params["weight"].get<double>();
		else
			energy_weights_[index] = 1.0;
	}

	double AMIPSEnergy::get_energy_weight(const int el_id) const
	{
		if (energy_weights_.empty())
			return 1.0;
		if (energy_weights_.size() == 1)
			return energy_weights_[0];
		if (el_id >= 0 && el_id < (int)energy_weights_.size())
			return energy_weights_[el_id];
		return 1.0;
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> AMIPSEnergy::gradient(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		const double det = polyfem::utils::determinant(F);
		if (det <= 0)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> grad(size(), size());
			grad.setConstant(std::nan(""));
			return grad;
		}

		const double weight = get_energy_weight(el_id);

		if (use_rest_pose_)
		{
			if (size() == 2)
				return weight * autogen::AMIPS2drest_gradient(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3drest_gradient(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				return weight * autogen::AMIPS2d_gradient(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3d_gradient(p, t, el_id, F);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> AMIPSEnergy::hessian(
		const RowVectorNd &p,
		const double t,
		const int el_id,
		const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3> &F) const
	{
		const double det = polyfem::utils::determinant(F);
		if (det <= 0)
		{
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 9, 9> hessian(size() * size(), size() * size());
			hessian.setConstant(std::nan(""));
			return hessian;
		}

		const double weight = get_energy_weight(el_id);

		if (use_rest_pose_)
		{
			if (size() == 2)
				return weight * autogen::AMIPS2drest_hessian(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3drest_hessian(p, t, el_id, F);
		}
		else
		{
			if (size() == 2)
				return weight * autogen::AMIPS2d_hessian(p, t, el_id, F);
			else
				return weight * autogen::AMIPS3d_hessian(p, t, el_id, F);
		}
	}
	/*
	void AMIPSEnergy::compute_scalar_value(
	const OutputData &data,
	std::vector<NamedMatrix> &result) const
	{
		const auto &local_pts = data.local_pts;
		const auto &bs = data.bs;
		const auto &gbs = data.gbs;
		const auto &displacement = data.fun;
		const int el_id = data.el_id;
		const double t = data.t;

		ElementAssemblyValues vals;
		vals.compute(el_id, size() == 3, local_pts, bs, gbs);

		Eigen::MatrixXd val(bs.bases.size(), 1);

		for (size_t j = 0; j < bs.bases.size(); ++j)
		{
			Eigen::MatrixXd deformation_grad(size(), size());
			compute_diplacement_grad(size(), vals, local_pts, j, displacement, deformation_grad);

			for (int d = 0; d < size(); ++d)
				deformation_grad(d, d) += 1;

			using DefGradMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;
			DefGradMat F = deformation_grad.cast<double>();

			const double e = elastic_energy<double>(local_pts.row(j), t, el_id, F);
			val(j, 0) = std::isfinite(e) ? e : 0.0;
		}

		result.push_back({"AMIPS", val});
	}*/

} // namespace polyfem::assembler