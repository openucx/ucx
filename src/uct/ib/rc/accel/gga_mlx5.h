/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.   
*                                                                             
* See file LICENSE for terms.                                                 
*/                                                                            
                                                                              
#ifndef UCT_GGA_MLX5_H_                                                       
#define UCT_GGA_MLX5_H_                                                       
                                                                              
#include <uct/ib/base/ib_md.h>                                                
                                                                              
                                                                              
ucs_status_t uct_ib_mlx5_gga_rkey_unpack(const uct_ib_md_packed_mkey_t *mkey, 
                                         uct_rkey_t *rkey_p, void **handle_p);
                                                                              
                                                                              
#endif /* UCT_GGA_MLX5_H_ */                                                  

