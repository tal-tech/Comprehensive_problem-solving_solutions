#pragma once

#include "clipper/clipper.hpp"
#include <iostream>

#include <vector>
#include <iterator>
#include <algorithm>
#include <iomanip>


// locality-aware NMS
namespace lanms {

	namespace cl = ClipperLib;

	struct Polygon {
		cl::Path poly;
		double score;
		void print_polygon(){
			for(auto iter = poly.begin(); iter != poly.end(); iter++)
			{
				std::cout << std::setprecision(11) << iter->X << " " << iter->Y << " ";
			}
			std::cout << score << std::endl;
		}
	};

	float paths_area(const ClipperLib::Paths &ps) {
		float area = 0;
		for (auto &&p: ps)
			area += cl::Area(p);
		return area;
	}

	float poly_iou(const Polygon &a, const Polygon &b) {
		cl::Clipper clpr;
		clpr.AddPath(a.poly, cl::ptSubject, true);
		clpr.AddPath(b.poly, cl::ptClip, true);

		cl::Paths inter, uni;
		clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
		clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);

		auto inter_area = paths_area(inter),
			 uni_area = paths_area(uni);
		return std::abs(inter_area) / std::max(std::abs(uni_area), 1.0f);
	}

	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold) {
		return poly_iou(a, b) > iou_threshold;
	}

	/**
	 * Incrementally merge polygons
	 */
	class PolyMerger {
		public:
			PolyMerger(): score(0), nr_polys(0) {
				memset(data, 0, sizeof(data));
			}

			/**
			 * Add a new polygon to be merged.
			 */
			void add(const Polygon &p_given) {
				Polygon p;
				if (nr_polys > 0) {
					// vertices of two polygons to merge may not in the same order;
					// we match their vertices by choosing the ordering that
					// minimizes the total squared distance.
					// see function normalize_poly for details.
					p = normalize_poly(get(), p_given);
				} else {
					p = p_given;
				}
				assert(p.poly.size() == 4);
				auto &poly = p.poly;
				auto s = p.score;
				data[0] += poly[0].X * s;
				data[1] += poly[0].Y * s;

				data[2] += poly[1].X * s;
				data[3] += poly[1].Y * s;

				data[4] += poly[2].X * s;
				data[5] += poly[2].Y * s;

				data[6] += poly[3].X * s;
				data[7] += poly[3].Y * s;

				score += p.score;

				nr_polys += 1;
			}

			inline std::int64_t sqr(std::int64_t x) { return x * x; }

			Polygon normalize_poly(
					const Polygon &ref,
					const Polygon &p) {

				std::int64_t min_d = std::numeric_limits<std::int64_t>::max();
				size_t best_start = 0, best_order = 0;

				for (size_t start = 0; start < 4; start ++) {
					size_t j = start;
					std::int64_t d = (
							sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 0) % 4].X)
							+ sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 0) % 4].Y)
							+ sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 1) % 4].X)
							+ sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 1) % 4].Y)
							+ sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 2) % 4].X)
							+ sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 2) % 4].Y)
							+ sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 3) % 4].X)
							+ sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 3) % 4].Y)
							);
					if (d < min_d) {
						min_d = d;
						best_start = start;
						best_order = 0;
					}

					d = (
							sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 3) % 4].X)
							+ sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 3) % 4].Y)
							+ sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 2) % 4].X)
							+ sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 2) % 4].Y)
							+ sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 1) % 4].X)
							+ sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 1) % 4].Y)
							+ sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 0) % 4].X)
							+ sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 0) % 4].Y)
						);
					if (d < min_d) {
						min_d = d;
						best_start = start;
						best_order = 1;
					}
				}

				Polygon r;
				r.poly.resize(4);
				auto j = best_start;
				if (best_order == 0) {
					for (size_t i = 0; i < 4; i ++)
						r.poly[i] = p.poly[(j + i) % 4];
				} else {
					for (size_t i = 0; i < 4; i ++)
						r.poly[i] = p.poly[(j + 4 - i - 1) % 4];
				}
				r.score = p.score;
				return r;
			}

			Polygon get() const {
				Polygon p;

				auto &poly = p.poly;
				poly.resize(4);
				auto score_inv = 1.0d / std::max(1e-8d, score);
				poly[0].X = data[0] * score_inv;
				poly[0].Y = data[1] * score_inv;
				poly[1].X = data[2] * score_inv;
				poly[1].Y = data[3] * score_inv;
				poly[2].X = data[4] * score_inv;
				poly[2].Y = data[5] * score_inv;
				poly[3].X = data[6] * score_inv;
				poly[3].Y = data[7] * score_inv;

				assert(score > 0);
				p.score = score;

				return p;
			}

		private:
			double data[8];
			double score;
			std::int32_t nr_polys;
	};


	/**
	 * The standard NMS algorithm.
	 */
	std::vector<Polygon> standard_nms(std::vector<Polygon> &polys, float iou_threshold) {
		size_t n = polys.size();
		if (n == 0)
			return {};
		std::vector<size_t> indices(n);
		std::iota(std::begin(indices), std::end(indices), 0);
		std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) { return polys[i].score > polys[j].score; });

		std::vector<size_t> keep;
		while (indices.size()) {
			size_t p = 0, cur = indices[0];
			keep.emplace_back(cur);
			for (size_t i = 1; i < indices.size(); i ++) {
				if (!should_merge(polys[cur], polys[indices[i]], iou_threshold)) {
					indices[p ++] = indices[i];
				}
			}
			indices.resize(p);
		}

		std::vector<Polygon> ret;
		for (auto &&i: keep) {
			ret.emplace_back(polys[i]);
		}
		return ret;
	}

	std::vector<Polygon>
		merge_quadrangle_n9(const double *data, size_t n, float iou_threshold) {
			using cInt = cl::cInt;

			// first pass
			std::vector<Polygon> polys;
			for (size_t i = 0; i < n; i ++) {
				auto p = data + i * 9;
				Polygon poly{
					{
						{cInt(p[0]), cInt(p[1])},
						{cInt(p[2]), cInt(p[3])},
						{cInt(p[4]), cInt(p[5])},
						{cInt(p[6]), cInt(p[7])},
					},
					p[8],
				};

				if (polys.size()) {
					// merge with the last one
					auto &bpoly = polys.back();
					if (should_merge(poly, bpoly, iou_threshold)) {
						PolyMerger merger;
						merger.add(bpoly);
						merger.add(poly);
						bpoly = merger.get();
					} else {
						polys.emplace_back(poly);
					}
				} else {
					polys.emplace_back(poly);
				}
			}
			return standard_nms(polys, iou_threshold);
		}
}
