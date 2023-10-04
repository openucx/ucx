/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_USAGE_TRACKER_H_
#define UCS_USAGE_TRACKER_H_

#include <stdint.h>

#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/linear_func.h>
#include <ucs/datastruct/lru.h>

/* Usage Tracker element data structure */
typedef struct {
    /* Key to use as hash table input */
    void  *key;
    /* Exponential decay based score ([0-1]) */
    double score;
    /* Lower bound to score value */
    double min_score;
    /* Is this entry considered 'promoted' */
    int    promoted;
} ucs_usage_tracker_element_t;


/* Callback type for rank modify notification */
typedef void (*ucs_usage_tracker_elem_update_cb_t)(void *entry, void *arg);


typedef struct {
    /* Max number of promoted entries */
    unsigned                           promote_capacity;
    /* Max number of entries to promote in each progress */
    unsigned                           promote_thresh;
    /* Min score difference in order to remove an entry from promoted list [0-1] */
    double                             remove_thresh;
    /* User callback which will be called when an entry is promoted */
    ucs_usage_tracker_elem_update_cb_t promote_cb;
    /* User object which will be passed to promote callback */
    void                              *promote_arg;
    /* User callback which will be called when an entry is demoted */
    ucs_usage_tracker_elem_update_cb_t demote_cb;
    /* User object which will be passed to demote callback */
    void                              *demote_arg;
    /* Exponential decay linear parameters (mult/add factors) */
    ucs_linear_func_t                  exp_decay;
} ucs_usage_tracker_params_t;


/* Hash table type for Usage Tracker class */
KHASH_INIT(usage_tracker_hash, uint64_t, ucs_usage_tracker_element_t, 1,
           kh_int64_hash_func, kh_int64_hash_equal);
typedef khash_t(usage_tracker_hash) ucs_usage_tracker_hash_t;


/* Usage Tracker main data structure */
typedef struct ucs_usage_tracker {
    /* Usage Tracker params */
    ucs_usage_tracker_params_t params;
    /* Hash table of addresses as keys */
    ucs_usage_tracker_hash_t   hash;
    /* LRU cache to track most active entries */
    ucs_lru_h                  lru;
} ucs_usage_tracker_t;


typedef struct ucs_usage_tracker *ucs_usage_tracker_h;


/**
 * @brief Create a new Usage Tracker object.
 *
 * Usage tracking is done by sampling an LRU cache of the most active entries.
 * Each sample we calculate the entries score based on exponential decay.
 * Entries with the high enough score will be promoted, while low score entries
 * will be demoted.
 *
 * @param [in]  params           usage tracker params.
 * @param [out] usage_tracker_p  Pointer to the Usage Tracker struct. Filled
 *                               with a Usage Tracker handle.
 *
 * @return UCS_OK if successful, or an error code as defined by
 * @ref ucs_status_t otherwise.
 */
ucs_status_t ucs_usage_tracker_create(const ucs_usage_tracker_params_t *params,
                                      ucs_usage_tracker_h *usage_tracker_p);


/**
 * @brief Destroys a Usage Tracker object.
 *
 * @param [in] usage_tracker  Handle to the Usage Tracker object.
 */
void ucs_usage_tracker_destroy(ucs_usage_tracker_h usage_tracker);


/**
 * @brief Progress the usage tracker.
 *
 * Triggers score update for all entries.
 * Calculates which entries should be promoted or demoted.
 *
 * @param [in] usage_tracker  Handle to the Usage Tracker object.
 */
void ucs_usage_tracker_progress(ucs_usage_tracker_h usage_tracker);


/**
 * @brief Update an entry with min score.
 *
 * @param [in]  usage_tracker  Handle to the Usage Tracker object.
 * @param [in]  key            Key to insert.
 * @param [in]  score          Min score of the entry.
 *
 */
void ucs_usage_tracker_set_min_score(ucs_usage_tracker_h usage_tracker,
                                     void *key, double score);


/**
 * @brief Get score of a specific entry.
 *
 * @param [in]  usage_tracker  Handle to the Usage Tracker object.
 * @param [in]  key            Key of the entry.
 * @param [out] score_p        Filled with the requested entry's score.
 *
 * @return UCS_OK if successful, or an error code as defined by
 * @ref ucs_status_t otherwise.
 */
ucs_status_t ucs_usage_tracker_get_score(ucs_usage_tracker_h usage_tracker,
                                         void *key, double *score_p);


/**
 * @brief Remove an entry from Usage Tracker.
 *
 * @param [in]  usage_tracker  Handle to the Usage Tracker object.
 * @param [in]  key            Key of the entry to be removed.
 *
 * @return UCS_OK if successful, or an error code as defined by
 * @ref ucs_status_t otherwise.
 */
ucs_status_t
ucs_usage_tracker_remove(ucs_usage_tracker_h usage_tracker, void *key);


/**
 * @brief Mark a key as being used.
 *
 * Track the usage of the given key in the usage tracker.
 * Should be called every time the key is being used.
 *
 * @param [in]  usage_tracker  Handle to the Usage Tracker object.
 * @param [in]  key            Key of the entry to be added.
 */
static UCS_F_ALWAYS_INLINE void
ucs_usage_tracker_touch_key(ucs_usage_tracker_h usage_tracker, void *key)
{
    ucs_lru_push(usage_tracker->lru, key);
}


/**
 * @brief Checks if a key is promoted.
 *
 * @param [in]  usage_tracker  Handle to the Usage Tracker object.
 * @param [in]  key            Key of the entry to be checked.
  
 * @return 1 if entry is promoted, 0 otherwise.
 */
int ucs_usage_tracker_is_promoted(ucs_usage_tracker_h usage_tracker, void *key);

#endif
