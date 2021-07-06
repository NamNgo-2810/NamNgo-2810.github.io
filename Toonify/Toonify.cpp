#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <vector>
#include <stack>
#include <map>
#include <cassert>
#include <limits>

using namespace cv;
using namespace std;

struct lessVec3b {
	bool operator()(Vec3b const& lhs, Vec3b const& rhs) const {
		return (lhs[2] != rhs[2]) ? (lhs[2] < rhs[2]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[0] < rhs[0]));
	}
};

typedef map<Vec3b, uint64_t, lessVec3b> map_type;

void reduce_colors_kmeans(Mat3b const& src, Mat3b& dst, int num_colors, vector<int>& labels, Mat1f& colors, map_type& palette) { // Giảm số màu trên từng vùng
	int n = src.rows * src.cols;
	Mat data = src.reshape(1, n);
	data.convertTo(data, CV_32F);

	labels.reserve(n);
	kmeans(data, num_colors, labels, TermCriteria(), 1, KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; i++) {
		int label = labels[i];
		data.at<float>(i, 0) = colors(label, 0);
		data.at<float>(i, 1) = colors(label, 1);
		data.at<float>(i, 2) = colors(label, 2);
		Vec3b color(colors(label, 0), colors(label, 1), colors(label, 2));
		auto it = palette.find(color);
		if (it == palette.end()) palette[color] = 1;
		else palette[color]++;
	}
	Mat reduced = data.reshape(3, src.rows);
	reduced.convertTo(dst, CV_8U);
}

void print_color_numbers(Mat3b& dst, int rows, int cols, vector<int> const& labels, vector<Point2f> const& centroids, Mat1f const& colors, map_type& palette) { // Xác định và tìm màu trong pallete, đánh số màu tương ứng với mỗi vùng
	static const float font_scale = 1.0;
	static const float thickness = 0.5;

	assert(labels.size() == centroids.size());

	static const size_t step = 1;
	for (size_t i = 0; i < centroids.size(); i += step) {
		if (i > centroids.size()) break;
		auto centroid = centroids[i];
		int label = labels[i];

		Vec3b color(colors(label, 0), colors(label, 1), colors(label, 2));

		auto it = palette.find(color);
		assert(it != palette.end());
		auto color_id = (*it).second;
		cout << "Region " << i << ": color_id " << color_id << endl;

		putText(dst, to_string(color_id), centroid, FONT_HERSHEY_PLAIN, font_scale, CV_RGB(255, 0, 0), thickness);
	}
}

void reshape2d(vector<int> const& labels, vector<vector<int>>& labels2d, int num_rows, int num_cols) {
	assert(labels.size() == num_rows * num_cols);
	labels2d.resize(num_rows, vector<int>());
	for (int i = 0; i != num_rows; i++) {
		labels2d[i].resize(num_cols);
		memcpy(&labels2d[i][0], &labels[i * num_cols], sizeof(int) * num_cols);
	}
}

bool is_valid(int r, int c, vector<vector<int>> const& labels2d) {
	return r >= 0 && r < labels2d.size() && c >= 0 && c < labels2d[0].size();
}

void visit(vector<vector<int>> const& labels2d, int r, int c, int label, vector<vector<bool>>& visited, vector<Point2f>& region) { // Duyệt DFS trên một vùng màu
	assert(r >= 0);
	assert(c >= 0);
	assert(r < labels2d.size());
	assert(c < labels2d[0].size());
	assert(labels2d[r][c] == label);

	if (visited[r][c]) return;

	stack<Point2f> st;
	vector<int> r_move = { -1, -1, -1, 0, 0, 1, 1, 1 };
	vector<int> c_move = { -1, 0, 1, -1, 1, -1, 0, 1 };
	st.emplace(r, c);

	while (!st.empty()) {
		auto point = st.top();
		st.pop();
		r = point.x;
		c = point.y;
		if (!visited[r][c]) {
			visited[r][c] = true;
			region.emplace_back(r, c);

			for (int i = 0; i < 8; i++) {
				if (is_valid(r + r_move[i], c + c_move[i], labels2d) && !visited[r + r_move[i]][c + c_move[i]] && labels2d[r + r_move[i]][c + c_move[i]] == label) {
					st.emplace(r + r_move[i], c + c_move[i]);
				}
			}
		}
	}
}

void visit(vector<vector<int>> const& labels2d, vector<vector<Point2f>>& regions) { // Tính tất cả các vùng màu
	int num_rows = labels2d.size();
	int num_cols = labels2d[0].size();

	vector<vector<bool>> visited(num_rows, vector<bool>(num_cols));

	for (int r = 0; r < num_rows; r++) {
		for (int c = 0; c < num_cols; c++) {
			vector<Point2f> region;
			int label = labels2d[r][c];
			visit(labels2d, r, c, label, visited, region); // Duyệt vùng màu cụ thể
			if (region.size() > 0) regions.push_back(region); // Thêm vào mảng regions
		}
	}
}

Point2f compute_centroid(vector<Point2f> const& region) {
	int K = 10;
	if (region.size() < K) K = region.size();
	const static int ITERATIONS = 3;
	vector<int> labels;
	vector<Point2f> centers;
	labels.reserve(region.size());
	kmeans(region, K, labels, TermCriteria(), ITERATIONS, KMEANS_PP_CENTERS, centers);

	vector<double> distances;
	distances.reserve(centers.size());

	for (auto p : centers) {
		double sum = 0.0;
		for (auto q : centers) {
			sum += norm(p - q);
			distances.push_back(sum);
		}
	}

	uint64_t index = min_element(distances.begin(), distances.end()) - distances.begin();
	assert(index < centers.size());
	auto center = centers[index];

	return { center.y, center.x };
}

// Sử dụng Canny để nhận diện nét
void auto_canny(Mat3b& src, Mat& canny, string const& filename) {
	Mat detected_edges;
	Canny(src, detected_edges, 1, 1, 3);
	canny = ~detected_edges;
}

int main() {
	string filename = "images\\caynui.jpg";
	int num_colors = 20;
	int blur_level = 21;
	Mat3b read = imread(filename);
	float scale_factor = 1.0;


	// Chỉnh lại kích cỡ ảnh theo scale factor
	Mat3b img;
	resize(read, img, Size(), scale_factor, scale_factor);


	// Làm mịn ảnh để giảm nhiễu
	Mat3b blurred;
	medianBlur(img, blurred, blur_level);

	// Áp dụng bilateral filtering
	Mat3b filtered;
	bilateralFilter(blurred, filtered, 9, 17, 17);

	// giảm số lượng màu
	Mat3b reduced;
	vector<int> labels;
	Mat1f colors;
	map_type palette;
	reduce_colors_kmeans(filtered, reduced, num_colors, labels, colors, palette);

	// Tính toán thang màu
	static const double MIN_AREA = 0.1;
	cout << "Rows: " << img.rows << endl;
	cout << "Cols: " << img.cols << endl;
	uint64_t area = img.rows * img.cols;

	uint64_t count = 0;
	for (auto& color : palette) {
		double color_area = (color.second * 100.0) / area;
		if (color_area > MIN_AREA) count++;
	}

	static const uint64_t PALETTE_SIZE = 150;
	static const uint64_t OFFSET = 50;

	// In các màu tính được và gán id từng màu
	Mat3b palette_img(PALETTE_SIZE + OFFSET, count * PALETTE_SIZE);
	palette_img.setTo(Scalar(255, 255, 255));
	uint64_t id = 1;

	for (auto& color : palette) {
		double color_area = (color.second * 100.0) / area;
		if (color_area > MIN_AREA) {
			color.second = id;
			cout << "Color " << color.second << ": rgb(" << int(color.first[2]) << ", " << int(color.first[1]) << ", " << int(color.first[0]) << ")" << " \t - Area: " << color_area << "%" << endl;

			uint64_t base = (id - 1) * PALETTE_SIZE;
			for (uint64_t i = 0; i < PALETTE_SIZE; i++) {
				for (uint64_t j = 0; j < PALETTE_SIZE; j++) {
					palette_img.at<Vec3b>(j + OFFSET, base + i) = color.first;
				}
			}

			putText(palette_img, to_string(id), Point(base + PALETTE_SIZE / 2 - 10, OFFSET / 2 + 10), FONT_HERSHEY_PLAIN, 2, CV_RGB(0, 0, 0), 2);
			id += 1;
		}
	}

	imwrite(filename + ".palette.jpeg", palette_img); // Ghi ra thang màu trong ảnh output

	vector<vector<int>> labels2d;
	reshape2d(labels, labels2d, img.rows, img.cols);

	// Tính toán diện tích các vùng màu
	vector<vector<Point2f>> regions;
	visit(labels2d, regions);
	cout << "Visit done!" << endl;

	// Nhận diện nét bằng Canny và hợp nhất thành kết quả
	Mat canny;
	auto_canny(reduced, canny, filename);
	imwrite(filename + ".canny.jpeg", canny); // Phiên bản nét của ảnh sau khi nhận diện bằng Canny
	imwrite(filename + ".toonified.jpeg", reduced); // Ảnh kết quả
	Mat3b canny3b = imread((filename + ".canny.jpeg").c_str());
	cout << "Done!" << endl;

	// Tính toán tâm của các vùng màu
	vector<Point2f> centroids;
	centroids.reserve(regions.size());
	vector<int> final_labels;
	final_labels.reserve(regions.size());

	uint64_t sum = 0;

	for (uint64_t i = 0; i < regions.size(); i++) {
		auto const& region = regions[i];
		sum += region.size();
		double color_area = (region.size() * 100.0) / area;
		if (color_area > MIN_AREA / 10) {
			auto centroid = compute_centroid(region);
			centroids.push_back(centroid);
			auto first_point = region.front();
			uint64_t index = first_point.x * img.cols + first_point.y;
			int label = labels[index];

			final_labels.push_back(label);
		}
	}

	assert(sum == area);
	cout << "#ignore " << sum << endl;

	print_color_numbers(canny3b, img.rows, img.cols, final_labels, centroids, colors, palette);
	imwrite(filename + "annotated.jpeg", canny3b); // Phiên bản nét được đánh số theo từng vùng

	return 0;
}