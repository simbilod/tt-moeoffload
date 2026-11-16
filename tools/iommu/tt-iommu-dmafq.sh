#!/bin/sh
# Put every Tenstorrent card's IOMMU group in DMA-FQ (translated DMA) mode. The engine's host link uses tt-metal's
# H2D/D2H sockets; with an identity group (the default under iommu=pt) tt-kmd pins only physically contiguous pages,
# so socket creation fails at random. Runs at boot from tt-iommu-dmafq.service and after every card reset through
# 99-tt-iommu-dmafq.rules (tt-smi -r re-enumerates the card and its group comes back identity); safe to rerun (a card
# already in DMA-FQ is left alone). It never unbinds a card in use: it waits up to 60 s for /dev/tenstorrent/* to be
# closed, then gives up with an error rather than pull the card from a job. The IOMMU itself must be on (BIOS, and
# amd_iommu / intel_iommu=on on the kernel command line); iommu=pt is fine, only the cards' groups change.
#
# Install (root):
#   sudo install -m 755 tools/iommu/tt-iommu-dmafq.sh /usr/local/sbin/
#   sudo install -m 644 tools/iommu/tt-iommu-dmafq.service /etc/systemd/system/
#   sudo systemctl daemon-reload && sudo systemctl enable --now tt-iommu-dmafq.service
#   sudo install -m 644 tools/iommu/99-tt-iommu-dmafq.rules /etc/udev/rules.d/ && sudo udevadm control --reload
# Revert: sudo systemctl disable tt-iommu-dmafq.service and remove the rule (the next boot is identity again).
set -e
DRV=/sys/bus/pci/drivers/tenstorrent
i=0
while ! ls "$DRV" 2>/dev/null | grep -q '^0000:' && [ $i -lt 60 ]; do   # wait for tt-kmd to bind the cards at boot
    sleep 1
    i=$((i + 1))
done
for path in "$DRV"/0000:*; do
    [ -e "$path" ] || continue
    DEV=$(basename "$path")
    GROUP=$(basename "$(readlink "/sys/bus/pci/devices/$DEV/iommu_group")")
    TYPE=/sys/kernel/iommu_groups/$GROUP/type
    [ "$(cat "$TYPE")" = DMA-FQ ] && continue
    j=0
    while fuser -s /dev/tenstorrent/* 2>/dev/null; do   # in use (tt-smi's re-initialisation, or a job): wait
        if [ $j -ge 60 ]; then
            logger -t tt-iommu-dmafq "$DEV still in use after 60 s; left as $(cat "$TYPE")"
            exit 1
        fi
        sleep 1
        j=$((j + 1))
    done
    echo "$DEV" > "$DRV/unbind"
    # the card is bound again whatever the type write does (EBUSY when another device in its group is bound)
    if ! echo DMA-FQ > "$TYPE"; then
        echo "$DEV" > "$DRV/bind"
        logger -t tt-iommu-dmafq "$DEV: group $GROUP refused DMA-FQ (another device bound in it?); left as $(cat "$TYPE")"
        exit 1
    fi
    echo "$DEV" > "$DRV/bind"
    logger -t tt-iommu-dmafq "$DEV (group $GROUP) now $(cat "$TYPE")"
    echo "tt-iommu-dmafq: $DEV (group $GROUP) now $(cat "$TYPE")"
done
