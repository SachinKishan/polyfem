#pragma once

#include <polyfem/mesh/RefinementMap.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace polyfem::mesh::detail
{
	/// Builds the RefinementMap between an NC mesh and a refinement of a copy of it.
	///
	/// NC meshes keep every vertex, edge and element they ever created in append-only lists
	/// (the "all" ids); refining only appends to them and never renumbers. The ids of the mesh
	/// interface (the "valid" ids) are recomputed from these lists by prepare_mesh(). A mesh
	/// copied from `coarse` and refined therefore shares the internal ids of `coarse`, and the
	/// correspondence is a lookup through them: valid -> all (coarse) -> valid (fine).
	///
	/// Both meshes must have had prepare_mesh() called after their last change.
	/// Instantiated for NCMesh2D and NCMesh3D, which declare this function a friend.
	template <class NC>
	RefinementMap build_nc_refinement_map(const NC &coarse, const NC &fine)
	{
		if (!coarse.index_prepared || !fine.index_prepared)
			throw std::runtime_error("refinement_map_to: call prepare_mesh() on both meshes first.");

		if (fine.vertices.size() < coarse.vertices.size()
			|| fine.edges.size() < coarse.edges.size()
			|| fine.elements.size() < coarse.elements.size())
			throw std::runtime_error("refinement_map_to: the second mesh is not a refinement of this mesh.");

		RefinementMap map;

		// vertices: same internal id, same position
		const int n_vertices = coarse.n_vertices();
		map.vertex.resize(n_vertices);
		for (int v = 0; v < n_vertices; ++v)
		{
			const int all = coarse.valid_to_all_vertexMap[v];
			const int fine_v = fine.all_to_valid_vertexMap[all];
			if (fine_v < 0)
				throw std::runtime_error("refinement_map_to: a vertex of the coarse mesh is missing in the fine mesh.");
			if ((coarse.vertices[all].pos - fine.vertices[all].pos).norm() > 1e-12)
				throw std::runtime_error("refinement_map_to: vertex positions differ, the second mesh is not a refinement of a copy of this mesh.");
			map.vertex[v] = fine_v;
		}

		// edges: the midpoint vertex is recorded by the mesh under the (sorted) pair of end vertices
		const int n_edges = coarse.n_edges();
		map.edge_midpoint.assign(n_edges, -1);
		for (int e = 0; e < n_edges; ++e)
		{
			const auto &ends = coarse.edges[coarse.valid_to_all_edgeMap[e]].vertices;
			const int mid_all = fine.find_vertex(ends(0), ends(1));
			if (mid_all >= 0)
				map.edge_midpoint[e] = fine.all_to_valid_vertexMap[mid_all];
		}

		// elements: follow the children down to the leaves, which are the valid elements
		const int n_elements = coarse.valid_to_all_elemMap.size();
		map.children.resize(n_elements);
		for (int e = 0; e < n_elements; ++e)
		{
			std::vector<int> stack{coarse.valid_to_all_elemMap[e]};
			while (!stack.empty())
			{
				const int all = stack.back();
				stack.pop_back();

				const auto &el = fine.elements[all];
				if (el.is_ghost)
					throw std::runtime_error("refinement_map_to: unexpected ghost element in the fine mesh.");

				if (!el.is_refined)
				{
					const int fine_e = fine.all_to_valid_elemMap[all];
					if (fine_e < 0)
						throw std::runtime_error("refinement_map_to: a leaf element has no valid id, call prepare_mesh().");
					map.children[e].push_back(fine_e);
				}
				else
				{
					for (int k = 0; k < el.children.size(); ++k)
					{
						if (el.children(k) < 0)
							throw std::runtime_error("refinement_map_to: refined element without children.");
						stack.push_back(el.children(k));
					}
				}
			}
			std::sort(map.children[e].begin(), map.children[e].end());
		}

		return map;
	}
} // namespace polyfem::mesh::detail
