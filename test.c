#include "reed_solomon_util.h"
#include "field_arithmetic_16.h"
#include "reed_solomon.h"

#include <stdio.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdbool.h>

struct TestParameters {
    unsigned original_count; // under 65536
    unsigned recovery_count; // = 2; under 65536 - original_count
    unsigned buffer_bytes; // multiple of 64 bytes
    unsigned loss_count; // = 2; some fraction of original_count
    unsigned seed; // = 2;
};

static bool test(struct TestParameters* params,char* filename);

static uint64_t GetTimeUsec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}


long long GetFileSize(char* filename){
    FILE *p_file = NULL;
    p_file = fopen(filename,"rb");
    if(p_file == NULL){
        printf("file not found");
        return -1;
    }
    fseek(p_file,0,SEEK_END);
    long long size = ftell(p_file);
    fclose(p_file);
    return size;
}

struct FunctionTimer {
    uint64_t t0;
    uint64_t Invokations;
    uint64_t TotalUsec;
    uint64_t MaxCallUsec;
    uint64_t MinCallUsec;

    void (*BeginCall)(struct FunctionTimer* this);
    void (*EndCall)(struct FunctionTimer* this);
    void (*Reset)(struct FunctionTimer* this);
 
};
extern const struct FunctionTimerClass{
    struct FunctionTimer (*new)(uint64_t t0,uint64_t Invokations,uint64_t TotalUsec,uint64_t MaxCallUsec,uint64_t MinCallUsec);
} FunctionTimer;

void BeginCall(struct FunctionTimer *this) {
    this->t0 = GetTimeUsec();
}

void EndCall(struct FunctionTimer *this) {
    const uint64_t t1 = GetTimeUsec();
    uint64_t delta = t1 - this->t0;
    if (++(this->Invokations) == 1)
       this-> MaxCallUsec = this->MinCallUsec = delta;
    else if (this->MaxCallUsec < delta)
        this->MaxCallUsec = delta;
    else if (this->MinCallUsec > delta)
        this->MinCallUsec = delta;
    this->TotalUsec += delta;
    this->t0 = 0;
}

void Reset(struct FunctionTimer *this) {
    this->t0 = 0;
    this->Invokations = 0;
    this->TotalUsec = 0;
}

static struct FunctionTimer new(uint64_t t0,uint64_t Invokations,uint64_t TotalUsec,uint64_t MaxCallUsec,uint64_t MinCallUsec){
     return (struct FunctionTimer){.t0=0,.Invokations=0,.MaxCallUsec=0,.MinCallUsec=0,.TotalUsec=0,.BeginCall=&BeginCall,.EndCall=&EndCall,.Reset=&Reset};   
}
const struct FunctionTimerClass FunctionTimer = {.new=&new};

static bool test(struct TestParameters* params,char* filename) {

    long long int total_bytes = GetFileSize(filename);
    // cout << total_bytes << endl;

    if(total_bytes % params->buffer_bytes == 0) 
        params->original_count = total_bytes/params->buffer_bytes;
    else 
        params->original_count = total_bytes/params->buffer_bytes + 1;

    // cout << params.original_count << endl;
    uint8_t* original_data[params->original_count];
    printf("%d",rs_encode_work_count(params->original_count, params->recovery_count));
    const unsigned encode_work_count = rs_encode_work_count(params->original_count, params->recovery_count);
    const unsigned decode_work_count = rs_decode_work_count(params->original_count, params->recovery_count);

    uint8_t* encode_work_data[encode_work_count];
    uint8_t* decode_work_data[decode_work_count];

    struct FunctionTimer t_mem_alloc = FunctionTimer.new(0,0,0,0,0);
    struct FunctionTimer fft_encode = FunctionTimer.new(0,0,0,0,0);
    struct FunctionTimer fft_decode = FunctionTimer.new(0,0,0,0,0);
    struct FunctionTimer t_mem_free = FunctionTimer.new(0,0,0,0,0);

    t_mem_alloc.BeginCall(&t_mem_alloc);
    for (unsigned i = 0, count = params->original_count; i < count; ++i)
        original_data[i] = SafeAllocate(params->buffer_bytes);
    for (unsigned i = 0, count = encode_work_count; i < count; ++i)
        encode_work_data[i] = SafeAllocate(params->buffer_bytes);
    for (unsigned i = 0, count = decode_work_count; i < count; ++i)
        decode_work_data[i] = SafeAllocate(params->buffer_bytes);
    t_mem_alloc.EndCall(&t_mem_alloc);

    FILE* data_file = fopen(filename,"rb");
    for(unsigned i = 0;i < params->original_count ; i++){
        fread(original_data[i],sizeof(char),params->buffer_bytes,data_file);
    }

    fft_encode.BeginCall(&fft_encode);
    ReedSolomonResult encodeResult = rs_encode(params->buffer_bytes, params->original_count, params->recovery_count, encode_work_count, (void**)&original_data[0], (void**)&encode_work_data[0]);
    fft_encode.EndCall(&fft_encode);

    if (encodeResult != success) {
        if (encodeResult == too_much_data) {
            printf("Skipping this test: Parameters are unsupported by the \n");
            return true;
        }
        // printf("Error:  encode failed with result= ");
        // printf(encodeResult << ": " << result_string(encodeResult) << endl;
        return false;
    }

    // create k packets of original data
        for(unsigned i = 1;i <= params->original_count ; i++){
            char* current_file[30];
            sprintf(current_file,"Coding/original_data_%d.mkv",i);
            FILE* file_pointer = NULL;
            file_pointer = fopen(current_file,"ab");
            fwrite(original_data[i-1],sizeof(char),params->buffer_bytes,file_pointer);
            fclose(file_pointer);
        }

        // create m packets of recovery dataS
        for(unsigned i = 1;i <= params->recovery_count ; i++){
            char* current_file[30];
            sprintf(current_file,"Coding/recovery_data_%d.mkv",i);
            FILE* file_pointer = NULL;
            file_pointer = fopen(current_file,"ab");
            fwrite(encode_work_data[i-1],sizeof(char),params->buffer_bytes,file_pointer);
            fclose(file_pointer);
        }

    unsigned actual_loss_count = 0;
    while(actual_loss_count < params->recovery_count){
        int file_no = rand()%params->original_count;
        char* delete_file[30];
        sprintf(delete_file,"Coding/original_data_%d.mkv",file_no);
        if(!remove(delete_file)) actual_loss_count++;
    }

    params->loss_count = actual_loss_count;

    for(unsigned i = 1;i <= params->original_count;i++){
        char* filename[30];
        sprintf(filename,"Coding/original_data_%d.mkv",i);
        if( access(filename,F_OK) != -1 ) {
            FILE* current_file = fopen(filename,"rb");
            fread(original_data[i-1],sizeof(char),params->buffer_bytes,current_file);
        } else {
            original_data[i-1] = NULL;
        }
    }

    for(unsigned i = 1;i <= params->recovery_count;i++){
        char* filename[30];
        sprintf(filename,"Coding/recovery_data_%d.mkv",i);
        if( access(filename,F_OK) != -1 ) {
            FILE* current_file = fopen(filename,"rb");
            fread(encode_work_data[i-1],sizeof(char),params->buffer_bytes,current_file);
            fclose(current_file);
        } else {
            encode_work_data[i-1] = NULL;
        }
    }


    fft_decode.BeginCall(&fft_decode);
    ReedSolomonResult decodeResult = rs_decode(total_bytes,params->buffer_bytes, params->original_count, params->recovery_count, decode_work_count, (void**)&original_data[0], (void**)&encode_work_data[0], (void**)&decode_work_data[0]);
    fft_decode.EndCall(&fft_decode);

    if (decodeResult != success) {
        // printf("Error: decode failed with result=" << decodeResult << ": " << result_string(decodeResult) << endl;
        return false;
    }    

    //create decoded file
    FILE* decoded_data = fopen("Coding/Decoded.mkv","ab");
    unsigned total = 0;
    for(unsigned i = 0;i < params->original_count;i++){
        if(original_data[i]){
            if(i == params->original_count - 1)
                fwrite(original_data[i],sizeof(char),total_bytes - total,decoded_data);
            else
                total += fwrite(original_data[i],sizeof(char),params->buffer_bytes,decoded_data);
            
        }else{
            if(i == params->original_count - 1)
                fwrite(decode_work_data[i],sizeof(char),total_bytes - total,decoded_data);
            else
                total += fwrite(decode_work_data[i],sizeof(char),params->buffer_bytes,decoded_data);   
        }
        
    }

    fclose(decoded_data);
        

    t_mem_free.BeginCall(&t_mem_free);
    for (unsigned i = 0, count = params->original_count; i < count; ++i)
        SafeFree(original_data[i]);
    for (unsigned i = 0, count = encode_work_count; i < count; ++i)
        SafeFree(encode_work_data[i]);
    for (unsigned i = 0, count = decode_work_count; i < count; ++i)
        SafeFree(decode_work_data[i]);
    t_mem_free.EndCall(&t_mem_free);

    float encode_MBPS = (total_bytes + ((uint64_t)params->buffer_bytes * params->recovery_count)) / (float)(fft_encode.MinCallUsec);
    float decode_MBPS = (total_bytes + ((uint64_t)params->buffer_bytes * params->recovery_count)) / (float)(fft_decode.MinCallUsec);

    printf("Parameters: [original count= %d ] ) [recovery count= %d ] [buffer bytes= %d ] [loss count= %d ]\n",params->original_count,params->recovery_count,params->buffer_bytes,params->loss_count);
    printf("Encoder( %f MB in %d pieces, %d losses) : Output = %f MB/s \n",(total_bytes*1.0)/1048576,params->original_count,params->loss_count,encode_MBPS);
    printf("Decoder( %f MB in %d pieces, %d losses) : Output = %f MB/s \n",(total_bytes*1.0)/1048576,params->original_count,params->loss_count,decode_MBPS);
    return true;
}

int main(int argc,char* argv[]) {
    struct TestParameters params;
    if(argc != 4){
        fprintf(stderr,"Insufficient arguments \n");
        fprintf(stderr,"argument 1: File Path \n");
        fprintf(stderr,"argument 2: Data packet size \n");
        fprintf(stderr,"argument 3: Number of Recovery packets \n");
        return 0;
    } 
    char* filename = argv[1];
    params.buffer_bytes = atoi(argv[2]);
    params.recovery_count = atoi(argv[3]);
    struct FunctionTimer rs_init = FunctionTimer.new(0,0,0,0,0);
    mkdir("Coding",0777);

    rs_init.BeginCall(&rs_init);
    rs_init_();
    rs_init.EndCall(&rs_init);

    if (params.loss_count > params.recovery_count)
        params.loss_count = params.recovery_count;

    test(&params,filename);

    return 0;
}
