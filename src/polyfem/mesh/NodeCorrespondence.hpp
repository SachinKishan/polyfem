#pragma once

#include <polyfem/mesh/MeshNodes.hpp>
#include <polyfem/mesh/RefinementMap.hpp>

#include <stdexcept>
#include <vector>

namespace polyfem::mesh
{
	/// For every node of the coarse space, the node of the fine space at the same position, or -1
	/// if the fine space has no such node (for example a coarse node that is a hanging node in
	/// the fine mesh). Node ids are the DOF ids, i.e. the rows of the solution vector of a scalar
	/// problem.
	///
	/// Only spaces whose nodes sit on mesh vertices and edge midpoints are supported, which is
	/// polynomial degree 1 and 2. For higher degrees there are face / cell nodes, and edge nodes
	/// that are not at the midpoint, and this function throws.
	inline std::vector<int> node_correspondence(
		const MeshNodes &coarse_nodes,
		const MeshNodes &fine_nodes,
		const RefinementMap &map)
	{
		if (coarse_nodes.num_face_nodes() > 0 || coarse_nodes.num_cell_nodes() > 0
			|| fine_nodes.num_face_nodes() > 0 || fine_nodes.num_cell_nodes() > 0)
			throw std::runtime_error("node_correspondence: only degree 1 and 2 spaces are supported.");

		// fine node of a fine vertex, -1 if that vertex has no node
		const auto fine_node_of_vertex = [&](const int fine_vertex) {
			if (fine_vertex < 0)
				return -1;
			return fine_nodes.primitive_to_node()[fine_nodes.primitive_from_vertex(fine_vertex)];
		};

		std::vector<int> result(coarse_nodes.n_nodes(), -1);
		for (int n = 0; n < coarse_nodes.n_nodes(); ++n)
		{
			const int primitive = coarse_nodes.node_to_primitive()[n];
			if (coarse_nodes.is_vertex_node(primitive))
				result[n] = fine_node_of_vertex(map.vertex[coarse_nodes.vertex_from_node_id(n)]);
			else if (coarse_nodes.is_edge_node(primitive))
				result[n] = fine_node_of_vertex(map.edge_midpoint[coarse_nodes.edge_from_node_id(n)]);
		}
		return result;
	}
} // namespace polyfem::mesh
