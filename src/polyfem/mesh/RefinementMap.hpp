#pragma once

#include <vector>

namespace polyfem::mesh
{
	/// Correspondence between a mesh and a mesh derived from it by refinement
	/// (the fine mesh is a copy of the coarse mesh, see Mesh::copy(), refined afterwards).
	/// All ids are the ids of the mesh interface (the "valid" ids), not the internal ones.
	struct RefinementMap
	{
		/// vertex[v]: vertex of the fine mesh at the same position as vertex v of the coarse mesh.
		std::vector<int> vertex;

		/// edge_midpoint[e]: vertex of the fine mesh at the midpoint of edge e of the coarse mesh,
		/// or -1 if the fine mesh has no such vertex.
		std::vector<int> edge_midpoint;

		/// children[c]: elements of the fine mesh that together cover element c of the coarse mesh.
		/// A coarse element that was not refined maps to itself.
		std::vector<std::vector<int>> children;
	};

	namespace detail
	{
		/// Shared implementation for NCMesh2D and NCMesh3D, defined in NCRefinementMap.hpp.
		template <class NC>
		RefinementMap build_nc_refinement_map(const NC &coarse, const NC &fine);
	} // namespace detail
} // namespace polyfem::mesh
