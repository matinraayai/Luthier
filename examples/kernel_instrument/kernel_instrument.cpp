#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <luthier.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <string>

static bool instrumented{false};

void check(bool pred) {
    if (!pred)
        std::exit(-1);
}

MARK_LUTHIER_DEVICE_MODULE

__managed__ int globalCounter = 20;

LUTHIER_DECLARE_FUNC void instrumentation_kernel(int* counter) {
    //    int i = 0;
    //    i = i + 4;
    //    i = i * 40;
    //    return i;
    //    return 1;
    *counter = *counter + 1;
    //    atomicAdd(counter, 1);
    //    printf("Hello from LUTHIER!\n");
}

LUTHIER_EXPORT_FUNC(instrumentation_kernel)


void luthier_at_init() {
    std::cout << "Kernel Instrument Tool is launching." << std::endl;
    luthier_enable_hsa_op_callback(HSA_API_ID_hsa_queue_create);
    luthier_enable_hsa_op_callback(HSA_API_ID_hsa_signal_store_screlease);
    luthier_enable_hsa_op_callback(HSA_API_ID_hsa_executable_freeze);
    luthier_enable_hsa_op_callback(HSA_EVT_ID_hsa_queue_packet_submit);
}

void luthier_at_term() {
    std::cout << "Counter Value: " << globalCounter << std::endl;
    std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}

void luthier_at_hsa_event(hsa_api_evt_args_t* args, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id) {
    if (phase == LUTHIER_API_EVT_PHASE_EXIT) {
        if (api_id == HSA_API_ID_hsa_queue_create) {
            std::cout << "Queue created called!" << std::endl;
            std::cout << "Signal handle: " << (*(args->api_args.hsa_queue_create.queue))->doorbell_signal.handle
                      << std::endl;
        } else if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
            fprintf(stdout, "<call to (%s)\t on %s> ", "hsa_signal_store_relaxed", "entry");
            std::cout << "Signal handle" << args->api_args.hsa_signal_store_relaxed.signal.handle << std::endl;

        } else if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->api_args.hsa_executable_freeze.executable;
            fprintf(stdout, "HSA Executable Freeze Callback\n");
            // Get the state of the executable (frozen or not frozen)
            hsa_executable_state_t e_state;
            check(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state) == HSA_STATUS_SUCCESS);

            fprintf(stdout, "Is executable frozen: %s\n", (e_state == HSA_EXECUTABLE_STATE_FROZEN ? "yes" : "no"));
            auto& coreTable = luthier_get_hsa_table()->core_;
            fprintf(stdout, "Executable handle: %lX\n", executable.handle);
        }
    }
    if (api_id == HSA_EVT_ID_hsa_queue_packet_submit) {
        std::cout << "In packet submission callback" << std::endl;
        auto packets = args->evt_args.hsa_queue_packet_submit.packets;
        for (unsigned int i = 0; i < args->evt_args.hsa_queue_packet_submit.pkt_count; i++) {
            auto& packet = packets[i];
            hsa_packet_type_t packetType = luthier_get_packet_type(&packet);

            if (packetType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
                std::cout << "Dispatch packet's kernel arg address: " << packet.dispatch.kernarg_address << std::endl;
                std::cout << "Size of private segment: " << packet.dispatch.private_segment_size << std::endl;
                packet.dispatch.private_segment_size = 100000;
                if (!instrumented) {
                    size_t instSize = 0;
                    luthier_disassemble_kernel_object(packet.dispatch.kernel_object, &instSize, nullptr);
                    std::vector<luthier_instruction_t> instrVec(instSize);
                    luthier_disassemble_kernel_object(packet.dispatch.kernel_object, &instSize, instrVec.data());
                    luthier_insert_call(instrVec[0], LUTHIER_GET_EXPORTED_FUNC(instrumentation_kernel),
                                        LUTHIER_IPOINT_AFTER);
                    instrumented = true;
                    luthier_override_with_instrumented(&packet.dispatch);
                }
            }
        }
        std::cout << "End of callback" << std::endl;
    }
}

void luthier_at_hip_event(void* args, luthier_api_evt_phase_t phase, int hip_api_id) {
    fprintf(stdout, "<call to (%s)\t on %s> ", hip_api_name(hip_api_id),
            phase == LUTHIER_API_EVT_PHASE_ENTER ? "entry" : "exit");
//    if (hip_api_id == HIP_API_ID_hipLaunchKernel) {
//        auto kern_args = reinterpret_cast<hip_hipLaunchKernel_api_args_t*>(args);
//        fprintf(stdout, "kernel(\"%s\") stream(%p)",
//                hipKernelNameRefByPtr(kern_args->function_address, kern_args->stream), kern_args->stream);
//    }
    fprintf(stdout, "\n");
    fflush(stdout);
}
