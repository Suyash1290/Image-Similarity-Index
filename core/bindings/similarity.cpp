#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern "C" {
    void neon_histogram(const uint8_t*, int, int, uint32_t*);
    void neon_phash(const uint8_t*, uint8_t*, int);
}

PYBIND11_MODULE(neon_ops, m) {
    m.def("neon_histogram", [](py::array_t<uint8_t> img) {
        auto buf = img.request();
        py::array_t<uint32_t> hist(256);
        auto hist_buf = hist.request();
        
        neon_histogram(static_cast<uint8_t*>(buf.ptr),
                       buf.shape[1],
                       buf.shape[0],
                       static_cast<uint32_t*>(hist_buf.ptr));
        
        return hist;
    });
    
    m.def("neon_phash", [](py::array_t<uint8_t> img) {
        auto buf = img.request();
        py::array_t<uint8_t> phash(64);
        auto phash_buf = phash.request();
        
        neon_phash(static_cast<uint8_t*>(buf.ptr),
                   static_cast<uint8_t*>(phash_buf.ptr),
                   buf.size);
        
        return phash;
    });
}
