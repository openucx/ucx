/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "libstats.h"
#include <ucs/sys/math.h>

#include <gtk/gtk.h>
#include <gtk/gtktreeview.h>
#include <stdint.h>
#include <inttypes.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


enum {
    COL_NAME  = 0,
    COL_VALUE = 1,
    COL_LOG_VALUE = 2,
    NUM_COLS
};


static ucs_stats_server_h      g_stats_server;
static GtkTreeStore            *g_treestore;
static GtkWidget               *g_treeview;


static void fill_model(ucs_stats_node_t *node, GtkTreeIter *parent, int depth)
{
    GtkTreeIter tree_elem, counter_elem;
    ucs_stats_counter_t value;
    ucs_stats_node_t *child;
    char buf[128];
    const char *name;
    double log_value;
    unsigned i;

    if (node == NULL) {
        return;
    }

    g_snprintf(buf, sizeof(buf), UCS_STATS_NODE_FMT, UCS_STATS_NODE_ARG(node));

    gtk_tree_store_append(g_treestore, &tree_elem, parent);
    gtk_tree_store_set(g_treestore, &tree_elem, COL_NAME, buf, -1);

    for (i = 0; i < node->cls->num_counters; ++i) {
        gtk_tree_store_append(g_treestore, &counter_elem, &tree_elem);

        name  = node->cls->counter_names[i];
        value = node->counters[i];

        g_snprintf(buf, sizeof(buf), "%" PRIu64, value);
        gtk_tree_store_set(g_treestore, &counter_elem,
                           COL_NAME,  name,
                           COL_VALUE, buf,
                           -1);

        if (value > 999) {
            log_value = ucs_log2(value);
            g_snprintf(buf, sizeof(buf), "%.2f", log_value);
            gtk_tree_store_set(g_treestore, &counter_elem,
                               COL_LOG_VALUE, buf,
                               -1);
        }
    }

    list_for_each(child, &node->children[UCS_STATS_ACTIVE_CHILDREN], list) {
        fill_model(child, &tree_elem, depth + 1);
    }
}

static gboolean update_view(GtkCellRenderer *renderer)
{
    ucs_stats_node_t *root;
    list_link_t *stats;
    GtkTreeModel *model;

    /* Get stats */
    stats = ucs_stats_server_get_stats(g_stats_server);

    /* Detach model from view */
    model = gtk_tree_view_get_model(GTK_TREE_VIEW(g_treeview));
    g_object_ref(model);
    gtk_tree_view_set_model(GTK_TREE_VIEW(g_treeview), NULL);

    /* Re-fill the model */
    gtk_tree_store_clear(g_treestore);
    list_for_each(root, stats, list) {
        fill_model(root, NULL, 0);
    }

    /* Re-attach model to view */
    gtk_tree_view_set_model(GTK_TREE_VIEW(g_treeview), model);
    g_object_unref(model);
    gtk_tree_view_expand_all(GTK_TREE_VIEW(g_treeview));

    /* Release stats */
    ucs_stats_server_purge_stats(g_stats_server);
    return TRUE;
}

static void name_cell_data_func (GtkTreeViewColumn *col,
                                 GtkCellRenderer   *renderer,
                                 GtkTreeModel      *model,
                                 GtkTreeIter       *iter,
                                 gpointer           user_data)
{
    if (gtk_tree_store_iter_depth(g_treestore, iter) == 0) {
        g_object_set(renderer,
                     "weight", PANGO_WEIGHT_BOLD,
                     "weight-set", TRUE,
                     "foreground", "blue",
                     "foreground-set", TRUE,
                     NULL);
    } else {
        g_object_set(renderer,
                     "weight-set", FALSE,
                     "foreground-set", FALSE,
                     NULL);
    }
}

 void value_cell_data_func (GtkTreeViewColumn *col,
                                  GtkCellRenderer   *renderer,
                                  GtkTreeModel      *model,
                                  GtkTreeIter       *iter,
                                  gpointer           user_data)
{
    gchar *value;

    gtk_tree_model_get(model, iter, COL_VALUE, &value, -1);

    if ((value == NULL) || (strcmp(value, "0") == 0)) {
        g_object_set(renderer,
                     "weight-set", FALSE,
                     NULL);
    } else {
        g_object_set(renderer,
                     "weight", PANGO_WEIGHT_BOLD,
                     "weight-set", TRUE,
                     NULL);
    }

    g_free(value);
}

static GtkWidget *create_tree_view(void)
{
    GtkTreeViewColumn *col;
    GtkCellRenderer *renderer;
    GtkWidget *view;

    view = gtk_tree_view_new();

    /* Name column */
    col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(col, "Counter");
    gtk_tree_view_append_column(GTK_TREE_VIEW(view), col);

    renderer = gtk_cell_renderer_text_new();
    gtk_tree_view_column_pack_start(col, renderer, TRUE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", COL_NAME);

    gtk_tree_view_column_set_min_width(col, 300);
    gtk_tree_view_column_set_cell_data_func(col, renderer, name_cell_data_func,
                                            NULL, NULL);

    /* Value column */
    col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(col, "Value");
    gtk_tree_view_append_column(GTK_TREE_VIEW(view), col);

    renderer = gtk_cell_renderer_text_new();
    gtk_tree_view_column_pack_start(col, renderer, TRUE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", COL_VALUE);

    gtk_tree_view_column_set_min_width(col, 200);
    gtk_tree_view_column_set_resizable(col, TRUE);
    gtk_tree_view_column_set_cell_data_func(col, renderer, value_cell_data_func,
                                            NULL, NULL);

    /* Log2 value column*/
    col = gtk_tree_view_column_new();
    gtk_tree_view_column_set_title(col, "Log2");
    gtk_tree_view_append_column(GTK_TREE_VIEW(view), col);

    renderer = gtk_cell_renderer_text_new();
    gtk_tree_view_column_pack_start(col, renderer, TRUE);
    gtk_tree_view_column_add_attribute(col, renderer, "text", COL_LOG_VALUE);

    gtk_tree_view_column_set_resizable(col, TRUE);
    gtk_tree_view_column_set_cell_data_func(col, renderer, value_cell_data_func,
                                            NULL, NULL);

    /* Disable selecting tree items */
    gtk_tree_selection_set_mode(gtk_tree_view_get_selection(GTK_TREE_VIEW(view)),
                                GTK_SELECTION_SINGLE);

    g_object_set(view,
                 "level-indentation", 20,
                 "show-expanders", FALSE,
                 NULL);

    gtk_tree_view_set_rules_hint(GTK_TREE_VIEW(view), TRUE);
    return view;
}

void usage() {
    printf("Usage: ucs_stats_reader [ -p <port> ] [ -d ]\n");
}

int main(int argc, char **argv)
{
    GtkWidget *window;
    ucs_status_t status;
    int port;
    int c;

    gtk_init(&argc, &argv);

    port    = UCS_STATS_DEFAULT_UDP_PORT;

    while ((c = getopt(argc, argv, "p:h")) != -1) {
        switch (c) {
        case 'p':
            port = atoi(optarg);
            break;
        case 'h':
        default:
            usage();
            return -1;
        }
    }

    status = ucs_stats_server_start(port, &g_stats_server);
    if (status != UCS_OK) {
        return -1;
    }

    /* Create window */
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
//    gtk_window_set_default_size(GTK_WINDOW(window), 300, 200);
//    gtk_window_set_position(GTK_WINDOW(window), GTK_WIN_POS_CENTER);
    g_signal_connect(window, "delete_event", gtk_main_quit, NULL); /* dirty */

    g_treestore = gtk_tree_store_new(NUM_COLS, G_TYPE_STRING, G_TYPE_STRING, G_TYPE_STRING);
    g_treeview = create_tree_view();
    gtk_tree_view_set_model(GTK_TREE_VIEW(g_treeview), GTK_TREE_MODEL(g_treestore));
    g_object_unref(g_treestore); /* destroy model automatically with view */

    gtk_container_add(GTK_CONTAINER(window), g_treeview);
    gtk_widget_show_all(window);

    g_timeout_add(500, (GSourceFunc) update_view, NULL);
    gtk_main();

    ucs_stats_server_destroy(g_stats_server);
    return 0;
}
