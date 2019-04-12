
def write_metrics_to_tb(metrics, writer, epoch, name):
    means = metrics.means()
    for metric, mean in means.items():
        writer.add_scalar('Train/{}_Mean_{}'.format(name, metric), mean, epoch)
