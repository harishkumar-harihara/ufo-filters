"""Generate burst laminographic backprojection OpenCL kernels."""
import argparse
import math
import os


IDX_TO_VEC_ELEM = dict(zip(range(10), range(10)))
IDX_TO_VEC_ELEM[10] = 'a'
IDX_TO_VEC_ELEM[11] = 'b'
IDX_TO_VEC_ELEM[12] = 'c'
IDX_TO_VEC_ELEM[13] = 'd'
IDX_TO_VEC_ELEM[14] = 'e'
IDX_TO_VEC_ELEM[15] = 'f'


def fill_compute_template(tmpl, num_items, index, constant):
    """Fill the template doing the pixel computation and texture fetch."""
    operation = '+' if index else ''
    if constant:
        access = '[{}]'.format(index)
    else:
        access = '.s{}'.format(IDX_TO_VEC_ELEM[index]) if num_items > 1 else ''

    return tmpl.format(index, access, operation)


def fill_kernel_template(input_tmpl, compute_tmpl, kernel_outer, kernel_inner, num_items,
                         constant):
    """Construct the whole kernel."""
    if constant:
        lut_str = 'constant float *sines,\nconstant float *cosines'
    else:
        vector_length = num_items if num_items > 1 else ''
        if vector_length:
            vector_length = int(2 ** math.ceil(math.log(num_items, 2)))
        lut_str = 'const float{0} sines,\nconst float{0} cosines'.format(vector_length)

    computes = '\n'.join([fill_compute_template(compute_tmpl, num_items, i, constant)
                          for i in range(num_items)])
    inputs = '\n'.join([input_tmpl.format(i) for i in range(num_items)])
    kernel_inner = kernel_inner.format(computes)

    return kernel_outer.format(num_items, inputs, lut_str, kernel_inner)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='File name with the kernel template')
    parser.add_argument('constant', type=int, choices=[0, 1],
                        help='Use constant memory to store sin and cos LUT')
    parser.add_argument('start', type=int,
                        help='Minimum number of projections processed by one kernel invocation')
    parser.add_argument('stop', type=int,
                        help='Maximum number of projections processed by one kernel invocation')

    args = parser.parse_args()
    if args.stop < args.start:
        raise ValueError("'stop' < 'start'")

    return args


def main():
    """execute program."""
    args = parse_args()
    in_tmpl = "read_only image2d_t projection_{},"
    common_filename = os.path.join(os.path.dirname(args.filename), 'common.in')
    defs_filename = os.path.join(os.path.dirname(args.filename), 'definitions.in')
    defs = open(defs_filename, 'r').read()
    kernel_outer = open(common_filename, 'r').read()
    comp_tmpl, kernel_inner = open(args.filename, 'r').read().split('\n%nl\n')
    kernels = defs + '\n'
    if args.stop > 16:
        raise ValueError("Maximum number of projections per invocation is 16")
    for burst in range(args.start, args.stop + 1):
        kernels += fill_kernel_template(in_tmpl, comp_tmpl, kernel_outer, kernel_inner, burst,
                                        args.constant)

    print kernels


if __name__ == '__main__':
    main()
