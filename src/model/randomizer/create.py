from src.model.classifier import GCN, GAT, APPNP, SAGE

def create_gnn(hparams):
    in_channels = hparams['in_channels']
    if hparams["smoothing_config"]["append_indicator"]:
            in_channels += 1

    if hparams['arch'] == "GAT":
        model = GAT(nfeat=in_channels, nlayers=1, nhid=2, heads=hparams['k_heads'], nclass=hparams['out_channels'], device=hparams['device'])
    elif hparams['arch'] == "GCN":
        model = GCN(nfeat=in_channels, nhid=hparams['hidden_channels'], nclass=hparams['out_channels'], device=hparams['device'])
    elif hparams['arch'] == "APPNP":
        model = APPNP(nfeat=in_channels, nhid=hparams['hidden_channels'], K=hparams['k_hops'], alpha=hparams['appnp_alpha'], nclass=hparams['out_channels'], device=hparams['device'])
    else:
        raise Exception("Not implemented")
    return model.to(hparams['device'])