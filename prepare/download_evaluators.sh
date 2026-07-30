#!/usr/bin/env bash
# Download the frozen evaluation encoders.
#
# These are needed for two things only: computing metrics, and the Stage-1
# perceptual loss. Text-to-motion generation and captioning do not touch them,
# so if you only want to run the released model you can skip this script.
#
# Required layout when this finishes (these are the paths the configs read):
#
#   deps/t2m/t2m/text_mot_match/model/finest.tar            HumanML3D evaluator
#   deps/t2m/t2m/Comp_v6_KLD005/meta/                       HumanML3D statistics
#   deps/glove/                                             word vectorizer
#   deps/snapmogen/evaluator/eval_*/model/                  SnapMoGen evaluator
#
# The archives below are distributed by the upstream projects and their internal
# folder layout is theirs, not ours. The verification step at the end tells you
# exactly what is still missing, so move the extracted folders into place if the
# archive nests them differently.
set -u

mkdir -p deps
cd deps

echo "==> HumanML3D evaluation models"
if [ ! -f humanml3d_evaluator.zip ]; then
    gdown --fuzzy https://drive.google.com/file/d/1sr73tfFk2O3-IL5brnZnylWi8oIVw_Hw/view?usp=drive_link \
        -O humanml3d_evaluator.zip
fi
unzip -n -q humanml3d_evaluator.zip && rm -f humanml3d_evaluator.zip

echo "==> SnapMoGen evaluation models"
if [ ! -f snapmogen_evaluator.zip ]; then
    gdown --fuzzy https://drive.google.com/file/d/1PfK_X_LuWz5rEZ__SXdUrZr-gxUbgUqc/view?usp=drive_link \
        -O snapmogen_evaluator.zip
fi
mkdir -p snapmogen
unzip -n -q snapmogen_evaluator.zip -d snapmogen && rm -f snapmogen_evaluator.zip

cd ..

echo
echo "==> Verifying the layout the configs expect"
missing=0
check() {
    if [ -e "$1" ]; then
        echo "    ok      $1"
    else
        echo "    MISSING $1"
        missing=1
    fi
}
check deps/t2m/t2m/text_mot_match/model/finest.tar
check deps/glove
check deps/snapmogen/evaluator

if [ "$missing" -ne 0 ]; then
    cat <<'EOF'

Some required files are not where the configs look for them. Move the extracted
folders so that the paths listed above exist, or edit configs/assets.yaml to
point at wherever you put them.

Upstream sources, if you would rather fetch them directly:
  HumanML3D evaluator and glove  https://github.com/EricGuo5513/HumanML3D
                                 (also shipped with MotionGPT's prepare scripts)
  SnapMoGen evaluator            https://github.com/snap-research/SnapMoGen
EOF
    exit 1
fi

echo
echo "Evaluation encoders are in place."
