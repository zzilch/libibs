#include <iostream>
#include <string>
#include <libqhullcpp/Qhull.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct IBS
{
	// input
	double *points;
	int *ids;
	// voro
	std::vector<double> vertices;
	std::vector<int> ridge_points;
	std::vector<std::vector<int>> ridge_vertices;
	// ibs
	std::vector<double> ibs_vertices;
	std::vector<int> ibs_points;
	std::vector<int> ibs_faces;
};

void find_ibs_ridges(qhT *qh, FILE *fp, vertexT *vertex, vertexT *vertexA, setT *centers, boolT unbounded)
{
	if (unbounded)
		return;

	IBS *ibs = (IBS *)fp;
	int point_1, point_2, idx_1, idx_2, ix;
	point_1 = qh_pointid(qh, vertex->point);
	point_2 = qh_pointid(qh, vertexA->point);
	idx_1 = ibs->ids[point_1];
	idx_2 = ibs->ids[point_2];

	if (idx_1 == idx_2)
		return;

	std::vector<int> cur_vertices;
	for (int i = 0; i < qh_setsize(qh, centers); i++)
	{
		ix = ((facetT *)centers->e[i].p)->visitid - 1;
		cur_vertices.push_back(ix);
	}

	if (cur_vertices.size() < 3)
		return;

	ibs->ridge_points.push_back(point_1);
	ibs->ridge_points.push_back(point_2);
	ibs->ridge_vertices.push_back(cur_vertices);
}

void find_ibs_vertices(qhT *qh, IBS &ibs)
{
	auto facet = qh->facet_list;
	int nv = 0;
	while (facet && facet->next)
	{
		if (facet->visitid > 0)
		{
			auto center = qh_facetcenter(qh, facet->vertices);

			int visitid = facet->visitid;
			nv = visitid > nv ? visitid : nv;
			if (nv >= ibs.vertices.size() / 3)
				ibs.vertices.resize((2 * nv + 1) * 3, 0);
			visitid -= 1;
			ibs.vertices[3 * visitid] = center[0];
			ibs.vertices[3 * visitid + 1] = center[1];
			ibs.vertices[3 * visitid + 2] = center[2];
			qh_memfree(qh, center, qh->center_size);
		}
		facet = facet->next;
	}
	ibs.vertices.resize(nv * 3);
}

void create_ibs_mesh(IBS &ibs)
{
	std::vector<int> new_vids(ibs.vertices.size() / 3, -1);
	int nridge = ibs.ridge_vertices.size();
	int nv = 0;
	for (int i = 0; i < nridge; i++)
	{
		for (int j = 0; j < ibs.ridge_vertices[i].size(); j++)
		{
			int vid = ibs.ridge_vertices[i][j];
			if (new_vids[vid] == -1)
			{
				new_vids[vid] = nv;
				ibs.ibs_vertices.push_back(ibs.vertices[3 * vid]);
				ibs.ibs_vertices.push_back(ibs.vertices[3 * vid + 1]);
				ibs.ibs_vertices.push_back(ibs.vertices[3 * vid + 2]);
				nv += 1;
			}
		}

		for (int j = 1; j < ibs.ridge_vertices[i].size() - 1; j++)
		{
			int vid0 = new_vids[ibs.ridge_vertices[i][0]];
			int vid1 = new_vids[ibs.ridge_vertices[i][j]];
			int vid2 = new_vids[ibs.ridge_vertices[i][j + 1]];

			int pid0 = ibs.ridge_points[2 * i];
			int pid1 = ibs.ridge_points[2 * i + 1];
			int idx0 = ibs.ids[pid0];
			int idx1 = ibs.ids[pid1];
			if (idx0 > idx1)
				std::swap(pid0, pid1);

			ibs.ibs_faces.push_back(vid0);
			ibs.ibs_faces.push_back(vid1);
			ibs.ibs_faces.push_back(vid2);
			ibs.ibs_points.push_back(pid0);
			ibs.ibs_points.push_back(pid1);
		}
	}
}

auto create_ibs(py::array_t<double> &points, py::array_t<int> &ids)
{
	IBS ibs;
	ibs.points = (double *)points.data(0);
	ibs.ids = (int *)ids.data(0);

	orgQhull::Qhull qhull;
	qhull.runQhull("", 3, ids.size(), ibs.points, "v");
	auto qh = qhull.qh();
	qh_eachvoronoi_all(qh, (FILE *)&ibs, find_ibs_ridges, qh->UPPERdelaunay, qh_RIDGEall, true);
	find_ibs_vertices(qh, ibs);
	create_ibs_mesh(ibs);

	//// return 2-D NumPy array
	py::array v = py::array(py::buffer_info(
		ibs.ibs_vertices.data(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		2,
		{int(ibs.ibs_vertices.size() / 3), 3},
		{sizeof(double) * 3, sizeof(double)}));

	py::array f = py::array(py::buffer_info(
		ibs.ibs_faces.data(),
		sizeof(int),
		py::format_descriptor<int>::format(),
		2,
		{int(ibs.ibs_faces.size() / 3), 3},
		{sizeof(int) * 3, sizeof(int)}));

	py::array p = py::array(py::buffer_info(
		ibs.ibs_points.data(),
		sizeof(int),
		py::format_descriptor<int>::format(),
		2,
		{int(ibs.ibs_points.size() / 2), 2},
		{sizeof(int) * 2, sizeof(int)}));

	return py::make_tuple(v, f, p);
}

void create_ibs_mesh_with_bounds(IBS &ibs, double *bounds)
{
	// auto in_bounds = [&](double *vptr) -> bool
	// {
	// 	return (vptr[0] >= bounds[0]) && (vptr[0] <= bounds[1]) && (vptr[1] >= bounds[2]) && (vptr[1] <= bounds[3]) && (vptr[2] >= bounds[4]) && (vptr[2] <= bounds[5]);
	// };
	auto in_bounds = [&](double *vptr) -> bool
	{
		return (vptr[0]-bounds[0])*(vptr[0]-bounds[0])+(vptr[1]-bounds[1])*(vptr[1]-bounds[1])+(vptr[2]-bounds[2])*(vptr[2]-bounds[2]) <= bounds[3]*bounds[3];
	};


	std::vector<int> new_vids(ibs.vertices.size() / 3, -1);
	int nridge = ibs.ridge_vertices.size();
	int nv = 0;
	for (int i = 0; i < nridge; i++)
	{
		for (int j = 0; j < ibs.ridge_vertices[i].size(); j++)
		{
			int vid = ibs.ridge_vertices[i][j];
			if ((new_vids[vid] == -1))
			{
				if (!in_bounds(&ibs.vertices[3 * vid]))
				{
					new_vids[vid] = -2;
				}
				else
				{
					new_vids[vid] = nv;
					ibs.ibs_vertices.push_back(ibs.vertices[3 * vid]);
					ibs.ibs_vertices.push_back(ibs.vertices[3 * vid + 1]);
					ibs.ibs_vertices.push_back(ibs.vertices[3 * vid + 2]);
					nv += 1;
				}
			}
		}

		for (int j = 1; j < ibs.ridge_vertices[i].size() - 1; j++)
		{
			int vid0 = new_vids[ibs.ridge_vertices[i][0]];
			int vid1 = new_vids[ibs.ridge_vertices[i][j]];
			int vid2 = new_vids[ibs.ridge_vertices[i][j + 1]];
			if (vid0 < 0 || vid1 < 0 || vid2 < 0)
				continue;
			;

			int pid0 = ibs.ridge_points[2 * i];
			int pid1 = ibs.ridge_points[2 * i + 1];
			int idx0 = ibs.ids[pid0];
			int idx1 = ibs.ids[pid1];
			if (idx0 > idx1)
				std::swap(pid0, pid1);

			ibs.ibs_faces.push_back(vid0);
			ibs.ibs_faces.push_back(vid1);
			ibs.ibs_faces.push_back(vid2);
			ibs.ibs_points.push_back(pid0);
			ibs.ibs_points.push_back(pid1);
			ibs.ibs_points.push_back(i);
		}
	}
}

auto create_ibs_with_bounds(py::array_t<double> &points, py::array_t<int> &ids, py::array_t<double> &bounds)
{
	IBS ibs;
	ibs.points = (double *)points.data(0);
	ibs.ids = (int *)ids.data(0);

	orgQhull::Qhull qhull;
	qhull.runQhull("", 3, ids.size(), ibs.points, "v");
	auto qh = qhull.qh();
	qh_eachvoronoi_all(qh, (FILE *)&ibs, find_ibs_ridges, qh->UPPERdelaunay, qh_RIDGEall, true);
	find_ibs_vertices(qh, ibs);
	create_ibs_mesh_with_bounds(ibs, (double *)bounds.data(0));

	//// return 2-D NumPy array
	py::array v = py::array(py::buffer_info(
		ibs.ibs_vertices.data(),
		sizeof(double),
		py::format_descriptor<double>::format(),
		2,
		{int(ibs.ibs_vertices.size() / 3), 3},
		{sizeof(double) * 3, sizeof(double)}));

	py::array f = py::array(py::buffer_info(
		ibs.ibs_faces.data(),
		sizeof(int),
		py::format_descriptor<int>::format(),
		2,
		{int(ibs.ibs_faces.size() / 3), 3},
		{sizeof(int) * 3, sizeof(int)}));

	py::array p = py::array(py::buffer_info(
		ibs.ibs_points.data(),
		sizeof(int),
		py::format_descriptor<int>::format(),
		2,
		{int(ibs.ibs_points.size() / 3), 3},
		{sizeof(int) * 3, sizeof(int)}));

	return py::make_tuple(v, f, p);
}

PYBIND11_MODULE(libibs, m)
{
	m.doc() = "IBS python wrapper";
	m.def("create_ibs", &create_ibs);
	m.def("create_ibs_with_bounds", &create_ibs_with_bounds);
}
