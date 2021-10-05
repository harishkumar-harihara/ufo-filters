/*
 * Copyright (C) 2011-2015 Karlsruhe Institute of Technology
 *
 * This file is part of Ufo.
 *
 * This library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "config.h"
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "ufo-stacked-forwardproject-task.h"
#include <stdio.h>

typedef enum {
    INT8,
    HALF,
    SINGLE
} Precision;

static GEnumValue precision_values[] = {
        {INT8,"INT8","int8"},
        {HALF, "HALF", "half"},
        {SINGLE, "SINGLE", "single"}
};

struct _UfoStackedForwardprojectTaskPrivate {
    cl_context context;
    cl_kernel interleave_single;
    cl_kernel texture_single;
    cl_kernel uninterleave_single;
    gfloat angle_step;
    gfloat axis_pos;
    guint num_projections;
    Precision precision;
};

static void ufo_task_interface_init (UfoTaskIface *iface);

G_DEFINE_TYPE_WITH_CODE (UfoStackedForwardprojectTask, ufo_stacked_forwardproject_task, UFO_TYPE_TASK_NODE,
                         G_IMPLEMENT_INTERFACE (UFO_TYPE_TASK,
                                                ufo_task_interface_init))

#define UFO_STACKED_FORWARDPROJECT_TASK_GET_PRIVATE(obj) (G_TYPE_INSTANCE_GET_PRIVATE((obj), UFO_TYPE_STACKED_FORWARDPROJECT_TASK, UfoStackedForwardprojectTaskPrivate))

enum {
    PROP_0,
    PROP_AXIS_POSITION,
    PROP_ANGLE_STEP,
    PROP_NUM_PROJECTIONS,
    PROP_PRECISION,
    N_PROPERTIES
};

static GParamSpec *properties[N_PROPERTIES] = { NULL, };

UfoNode *
ufo_stacked_forwardproject_task_new (void)
{
    return UFO_NODE (g_object_new (UFO_TYPE_STACKED_FORWARDPROJECT_TASK, NULL));
}

static void
ufo_stacked_forwardproject_task_setup (UfoTask *task,
                       UfoResources *resources,
                       GError **error)
{
    UfoStackedForwardprojectTaskPrivate *priv;

    priv = UFO_STACKED_FORWARDPROJECT_TASK(task)->priv;

    priv->context = ufo_resources_get_context (resources);
    priv->interleave_single = ufo_resources_get_kernel (resources, "stacked-forwardproject.cl", "interleave_single", NULL, error);
    priv->texture_single = ufo_resources_get_kernel (resources, "stacked-forwardproject.cl", "texture_single", NULL, error);
    priv->uninterleave_single = ufo_resources_get_kernel (resources, "stacked-forwardproject.cl", "uninterleave_single", NULL, error);

    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainContext (priv->context), error);

    if (priv->interleave_single != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->interleave_single), error);

    if (priv->texture_single != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->texture_single), error);

    if (priv->uninterleave_single != NULL)
    UFO_RESOURCES_CHECK_SET_AND_RETURN (clRetainKernel (priv->uninterleave_single), error);

    if (priv->angle_step == 0)
        priv->angle_step = G_PI / priv->num_projections;
}

static void
ufo_stacked_forwardproject_task_get_requisition (UfoTask *task,
                                 UfoBuffer **inputs,
                                 UfoRequisition *requisition,
                                 GError **error)
{
    UfoStackedForwardprojectTaskPrivate *priv;
    UfoRequisition in_req;

    priv = UFO_STACKED_FORWARDPROJECT_TASK(task)->priv;

    ufo_buffer_get_requisition (inputs[0], &in_req);

    requisition->n_dims = 3;
    requisition->dims[0] = in_req.dims[0];
    requisition->dims[1] = priv->num_projections;
    requisition->dims[2] = in_req.dims[2];
    if (priv->axis_pos == -G_MAXFLOAT) {
        priv->axis_pos = in_req.dims[0] / 2.0f;
    }

}

static guint
ufo_stacked_forwardproject_task_get_num_inputs (UfoTask *task)
{
    return 1;
}

static guint
ufo_stacked_forwardproject_task_get_num_dimensions (UfoTask *task,
                                             guint input)
{
    g_return_val_if_fail (input == 0, 0);
    return 3;
}

static UfoTaskMode
ufo_stacked_forwardproject_task_get_mode (UfoTask *task)
{
    return UFO_TASK_MODE_PROCESSOR | UFO_TASK_MODE_GPU;
}

static gboolean
ufo_stacked_forwardproject_task_process (UfoTask *task,
                         UfoBuffer **inputs,
                         UfoBuffer *output,
                         UfoRequisition *requisition)
{
    UfoStackedForwardprojectTaskPrivate *priv;
    UfoGpuNode *node;
    UfoProfiler *profiler;
    cl_command_queue cmd_queue;
    cl_mem interleaved_img;
    cl_mem out_mem;
    cl_mem reconstructed_buffer;
    cl_mem device_array;

    cl_kernel kernel_interleave;
    cl_kernel kernel_texture;
    cl_kernel kernel_uninterleave;

    size_t buffer_size;

    priv = UFO_STACKED_FORWARDPROJECT_TASK (task)->priv;
    node = UFO_GPU_NODE (ufo_task_node_get_proc_node (UFO_TASK_NODE (task)));
    cmd_queue = ufo_gpu_node_get_cmd_queue (node);
    out_mem = ufo_buffer_get_device_array (output, cmd_queue);
    profiler = ufo_task_node_get_profiler (UFO_TASK_NODE (task));

    // Image format
    cl_image_format format;
    device_array = ufo_buffer_get_device_array(inputs[0],cmd_queue);

    UfoRequisition req;
    ufo_buffer_get_requisition(inputs[0],&req);

    unsigned long dim_x = (requisition->dims[0]%16 == 0) ? requisition->dims[0] : (((requisition->dims[0]/16)+1)*16);
    unsigned long dim_y = (requisition->dims[1]%16 == 0) ? requisition->dims[1] : (((requisition->dims[1]/16)+1)*16);
    unsigned long quotient;

    if(priv->precision == SINGLE){
        quotient = requisition->dims[2]/2;
        kernel_interleave = priv->interleave_single;
        kernel_texture = priv->texture_single;
        kernel_uninterleave = priv->uninterleave_single;
        format.image_channel_order = CL_RG;
        format.image_channel_data_type = CL_FLOAT;
        buffer_size = sizeof(cl_float2) * dim_x * dim_y * quotient;
    }

    cl_image_desc imageDesc;
    imageDesc.image_width = req.dims[0];
    imageDesc.image_height = req.dims[1];
    imageDesc.image_depth = 0;
    imageDesc.image_array_size = quotient;
    imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    imageDesc.image_slice_pitch = 0;
    imageDesc.image_row_pitch = 0;
    imageDesc.num_mip_levels = 0;
    imageDesc.num_samples = 0;
    imageDesc.buffer = NULL;

    if(quotient > 0){
        // Interleave
        interleaved_img = clCreateImage(priv->context, CL_MEM_READ_WRITE, &format, &imageDesc, NULL, 0);
        UFO_RESOURCES_CHECK_CLERR(clSetKernelArg(kernel_interleave, 0, sizeof(cl_mem), &device_array));
        UFO_RESOURCES_CHECK_CLERR(clSetKernelArg(kernel_interleave, 1, sizeof(cl_mem), &interleaved_img));
        size_t gsize_interleave[3] = {req.dims[0],req.dims[1],quotient};
        ufo_profiler_call(profiler, cmd_queue, kernel_interleave, 3, gsize_interleave, NULL);

        //Forward projection
        reconstructed_buffer = clCreateBuffer(priv->context, CL_MEM_READ_WRITE, buffer_size, NULL, 0);
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel_texture, 0, sizeof (cl_mem), &interleaved_img));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel_texture, 1, sizeof (cl_mem), &reconstructed_buffer));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel_texture, 2, sizeof (gfloat), &priv->axis_pos));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel_texture, 3, sizeof (gfloat), &priv->angle_step));
        UFO_RESOURCES_CHECK_CLERR (clSetKernelArg (kernel_texture, 4, sizeof(unsigned long), &requisition->dims[0]));
        size_t gsize_texture[3] = {dim_x,dim_y,quotient};
        size_t lSize[3] = {16,16,1};
        ufo_profiler_call(profiler, cmd_queue, kernel_texture, 3, gsize_texture, lSize);

        //Uninterleave
        UFO_RESOURCES_CHECK_CLERR(clSetKernelArg(kernel_uninterleave, 0, sizeof(cl_mem), &reconstructed_buffer));
        UFO_RESOURCES_CHECK_CLERR(clSetKernelArg(kernel_uninterleave, 1, sizeof(cl_mem), &out_mem));
        size_t gsize_uninterleave[3] = {requisition->dims[0],requisition->dims[1],quotient};
        ufo_profiler_call(profiler, cmd_queue, kernel_uninterleave, 3, gsize_uninterleave, NULL);

        UFO_RESOURCES_CHECK_CLERR(clReleaseMemObject(interleaved_img));
        UFO_RESOURCES_CHECK_CLERR(clReleaseMemObject(reconstructed_buffer));
    }

    return TRUE;
}


static void
ufo_stacked_forwardproject_task_set_property (GObject *object,
                              guint property_id,
                              const GValue *value,
                              GParamSpec *pspec)
{
    UfoStackedForwardprojectTaskPrivate *priv = UFO_STACKED_FORWARDPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_AXIS_POSITION:
            priv->axis_pos = g_value_get_float (value);
            break;
        case PROP_ANGLE_STEP:
            priv->angle_step = g_value_get_float(value);
            break;
        case PROP_NUM_PROJECTIONS:
            priv->num_projections = g_value_get_uint(value);
            break;
        case PROP_PRECISION:
            priv->precision = g_value_get_enum(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_stacked_forwardproject_task_get_property (GObject *object,
                              guint property_id,
                              GValue *value,
                              GParamSpec *pspec)
{
    UfoStackedForwardprojectTaskPrivate *priv = UFO_STACKED_FORWARDPROJECT_TASK_GET_PRIVATE (object);

    switch (property_id) {
        case PROP_AXIS_POSITION:
            g_value_set_float (value, priv->axis_pos);
            break;
        case PROP_ANGLE_STEP:
            g_value_set_float(value, priv->angle_step);
            break;
        case PROP_NUM_PROJECTIONS:
            g_value_set_uint(value, priv->num_projections);
            break;
        case PROP_PRECISION:
            g_value_set_enum(value,priv->precision);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
            break;
    }
}

static void
ufo_stacked_forwardproject_task_finalize (GObject *object)
{
    UfoStackedForwardprojectTaskPrivate *priv;

    priv = UFO_STACKED_FORWARDPROJECT_TASK_GET_PRIVATE (object);

    if (priv->interleave_single) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->interleave_single));
        priv->interleave_single = NULL;
    }

    if (priv->texture_single) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->texture_single));
        priv->texture_single = NULL;
    }

    if (priv->uninterleave_single) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseKernel (priv->uninterleave_single));
        priv->uninterleave_single = NULL;
    }

    if (priv->context) {
        UFO_RESOURCES_CHECK_CLERR (clReleaseContext (priv->context));
        priv->context = NULL;
    }

    G_OBJECT_CLASS (ufo_stacked_forwardproject_task_parent_class)->finalize (object);
}

static void
ufo_task_interface_init (UfoTaskIface *iface)
{
    iface->setup = ufo_stacked_forwardproject_task_setup;
    iface->get_num_inputs = ufo_stacked_forwardproject_task_get_num_inputs;
    iface->get_num_dimensions = ufo_stacked_forwardproject_task_get_num_dimensions;
    iface->get_mode = ufo_stacked_forwardproject_task_get_mode;
    iface->get_requisition = ufo_stacked_forwardproject_task_get_requisition;
    iface->process = ufo_stacked_forwardproject_task_process;
}

static void
ufo_stacked_forwardproject_task_class_init (UfoStackedForwardprojectTaskClass *klass)
{
    GObjectClass *oclass = G_OBJECT_CLASS (klass);

    oclass->set_property = ufo_stacked_forwardproject_task_set_property;
    oclass->get_property = ufo_stacked_forwardproject_task_get_property;
    oclass->finalize = ufo_stacked_forwardproject_task_finalize;

    properties[PROP_AXIS_POSITION] =
            g_param_spec_float ("axis-pos",
                                "Position of rotation axis",
                                "Position of rotation axis",
                                -G_MAXFLOAT, G_MAXFLOAT, -G_MAXFLOAT,
                                G_PARAM_READWRITE);

    properties[PROP_ANGLE_STEP] =
            g_param_spec_float("angle-step",
                               "Increment of angle in radians",
                               "Increment of angle in radians",
                               -4.0f * ((gfloat) G_PI),
                               +4.0f * ((gfloat) G_PI),
                               0.0f,
                               G_PARAM_READWRITE);

    properties[PROP_NUM_PROJECTIONS] =
            g_param_spec_uint("number",
                              "Number of projections",
                              "Number of projections",
                              1, 32768, 256,
                              G_PARAM_READWRITE);

    properties[PROP_PRECISION] =
            g_param_spec_enum("precision-mode",
                              "Precision mode (\"int8\", \"half\", \"single\")",
                              "Precision mode (\"int8\", \"half\", \"single\")",
                              g_enum_register_static("ufo_stacked_forwardproject_precision", precision_values),
                              SINGLE, G_PARAM_READWRITE);

    for (guint i = PROP_0 + 1; i < N_PROPERTIES; i++)
        g_object_class_install_property (oclass, i, properties[i]);

    g_type_class_add_private (oclass, sizeof(UfoStackedForwardprojectTaskPrivate));
}

static void
ufo_stacked_forwardproject_task_init(UfoStackedForwardprojectTask *self)
{
    self->priv = UFO_STACKED_FORWARDPROJECT_TASK_GET_PRIVATE(self);

    self->priv->axis_pos = -G_MAXFLOAT;
    self->priv->num_projections = 256;
    self->priv->angle_step = 0;
    self->priv->precision = SINGLE;
}
