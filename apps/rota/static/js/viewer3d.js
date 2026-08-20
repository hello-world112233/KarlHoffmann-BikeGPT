import * as THREE from "/static/vendor/three/build/three.module.js";
import { OrbitControls } from "/static/vendor/three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "/static/vendor/three/examples/jsm/loaders/GLTFLoader.js";

/** Visible body limbs (skip eye/ear clutter). */
const LIMBS = [
  { a: 5, b: 7, ra: 0.052, rb: 0.037, kind: "upperArm" },
  { a: 7, b: 9, ra: 0.038, rb: 0.025, kind: "forearm" },
  { a: 6, b: 8, ra: 0.052, rb: 0.037, kind: "upperArm" },
  { a: 8, b: 10, ra: 0.038, rb: 0.025, kind: "forearm" },
  { a: 11, b: 13, ra: 0.068, rb: 0.051, kind: "thigh" },
  { a: 13, b: 15, ra: 0.049, rb: 0.029, kind: "shin" },
  { a: 12, b: 14, ra: 0.068, rb: 0.051, kind: "thigh" },
  { a: 14, b: 16, ra: 0.049, rb: 0.029, kind: "shin" },
];

function setTube(mesh, a, b) {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  if (len < 1e-4) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.scale.set(1, len, 1);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
}

function v3(x, y, z) {
  return new THREE.Vector3(x, y, z);
}

export class RotaViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.frames = [];
    this.idx = 0;
    this.playing = true;
    this._last = 0;
    this.fps = 10;
    this.onFrame = null;
    this._smooth = null;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x0b1410, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.05, 40);
    // Classic coaching side view
    this.camera.position.set(0.35, 0.85, -2.8);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.target.set(0.2, 0.35, 0);
    this.controls.minDistance = 1.4;
    this.controls.maxDistance = 6;

    this.scene.add(new THREE.HemisphereLight(0xf2ffe8, 0x1a2820, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 1.05);
    key.position.set(2.2, 3.5, 2.0);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xb8f06e, 0.35);
    rim.position.set(-2.5, 1.2, -1.5);
    this.scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(2.2, 72),
      new THREE.MeshStandardMaterial({ color: 0x121a16, roughness: 0.95, metalness: 0.02 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    this.scene.add(floor);
    this.floor = floor;
    const grid = new THREE.GridHelper(4.0, 20, 0x2c4034, 0x1a2a22);
    grid.position.y = 0.002;
    this.scene.add(grid);
    this.grid = grid;

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.bikeGroup = new THREE.Group();
    this.skelGroup = new THREE.Group();
    this.characterGroup = new THREE.Group();
    this.root.add(this.bikeGroup);
    this.root.add(this.skelGroup);
    this.root.add(this.characterGroup);

    this.bikeIsCalibrated = false;
    this._buildBike();
    this._buildHuman();

    window.addEventListener("resize", () => this.resize());
    this.resize();
    this._tick = this._tick.bind(this);
    requestAnimationFrame(this._tick);
  }

  _mat(color, opts = {}) {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: opts.roughness ?? 0.45,
      metalness: opts.metalness ?? 0.15,
      transparent: !!opts.opacity && opts.opacity < 1,
      opacity: opts.opacity ?? 1,
    });
  }

  _addTube(parent, r, color, metal = 0.35) {
    const m = new THREE.Mesh(
      new THREE.CylinderGeometry(r, r, 1, 14),
      this._mat(color, { metalness: metal, roughness: 0.35 })
    );
    parent.add(m);
    return m;
  }

  _addTaperedTube(parent, radiusA, radiusB, material) {
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(radiusB, radiusA, 1, 18, 1, false),
      material
    );
    parent.add(mesh);
    return mesh;
  }

  _clearBike() {
    while (this.bikeGroup.children.length) {
      const child = this.bikeGroup.children[0];
      this.bikeGroup.remove(child);
      child.traverse?.((item) => {
        item.geometry?.dispose?.();
        if (Array.isArray(item.material)) item.material.forEach((m) => m.dispose?.());
        else item.material?.dispose?.();
      });
    }
  }

  /** Procedural road bike driven by the five-point calibration. */
  _buildBike(geometry = null) {
    this._clearBike();
    const carbon = this._mat(0x202724, { metalness: 0.38, roughness: 0.3 });
    const carbonSoft = this._mat(0x303936, { metalness: 0.24, roughness: 0.42 });
    const rubber = this._mat(0x090b0a, { metalness: 0.02, roughness: 0.86 });
    const alloy = this._mat(0x9ba7a1, { metalness: 0.82, roughness: 0.24 });
    const accent = this._mat(0xa9df61, { metalness: 0.22, roughness: 0.36 });

    const point = (name, fallback) => {
      const value = geometry?.[name];
      return value && value.length >= 2 ? v3(value[0], value[1], value[2] || 0) : fallback;
    };
    this.bikeLocal = {
      bb: point("bottom_bracket", v3(0, 0, 0)),
      seat: point("saddle", v3(-0.12, 0.52, 0)),
      rearHub: point("rear_hub", v3(-0.42, -0.02, 0)),
      frontHub: point("front_hub", v3(0.62, -0.02, 0)),
      bar: point("handlebar", v3(0.58, 0.50, 0)),
      wheelR: Number(geometry?.wheel_radius || 0.33),
    };
    const L = this.bikeLocal;
    // Inferred structural joints give the mesh real bike topology without
    // asking the user to label more than the five semantic anchors.
    L.seatCluster = L.seat.clone().add(v3(0, -0.055, 0));
    L.headTop = L.bar.clone().lerp(L.frontHub, 0.31);
    L.headBottom = L.bar.clone().lerp(L.frontHub, 0.43);
    this._crankLen = Number(geometry?.crank_length || 0.17);
    this.bikeIsCalibrated = !!geometry;

    const mk = (a, b, r = 0.016, material = carbon) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 18), material);
      setTube(mesh, a, b);
      this.bikeGroup.add(mesh);
      return mesh;
    };
    // Main triangle, rear stays and a two-legged fork.
    mk(L.bb, L.seatCluster, 0.025);
    mk(L.bb, L.headBottom, 0.028);
    mk(L.seatCluster, L.headTop, 0.021);
    mk(L.headBottom, L.headTop, 0.025, accent);
    for (const z of [-0.035, 0.035]) {
      mk(L.bb.clone().add(v3(0, 0, z)), L.rearHub.clone().add(v3(0, 0, z)), 0.011);
      mk(L.seatCluster.clone().add(v3(0, 0, z)), L.rearHub.clone().add(v3(0, 0, z)), 0.010);
      mk(L.headBottom.clone().add(v3(0, 0, z)), L.frontHub.clone().add(v3(0, 0, z)), 0.012);
    }

    // Seatpost and softly rounded saddle.
    mk(L.seatCluster, L.seat, 0.014, alloy);
    const saddle = new THREE.Mesh(new THREE.SphereGeometry(0.1, 22, 14), rubber);
    saddle.scale.set(1.35, 0.23, 0.72);
    saddle.position.copy(L.seat);
    saddle.rotation.z = -0.08;
    this.bikeGroup.add(saddle);

    // Stem and road drop bar. The bar axis is lateral (Z), not frame-longitudinal.
    const stemEnd = L.bar.clone().add(v3(-0.035, -0.02, 0));
    mk(L.headTop, stemEnd, 0.013, alloy);
    const barWidth = Number(geometry?.handlebar_half_width || 0.18);
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, barWidth * 2, 16), carbonSoft);
    bar.rotation.x = Math.PI / 2;
    bar.position.copy(L.bar);
    this.bikeGroup.add(bar);
    for (const s of [-1, 1]) {
      const z = s * barWidth;
      const hood = new THREE.Mesh(new THREE.CapsuleGeometry(0.022, 0.055, 5, 10), rubber);
      hood.position.set(L.bar.x + 0.025, L.bar.y - 0.018, z);
      hood.rotation.z = Math.PI / 2.7;
      this.bikeGroup.add(hood);
      const drop = new THREE.Mesh(
        new THREE.TorusGeometry(0.065, 0.011, 10, 24, Math.PI * 1.35),
        carbonSoft
      );
      drop.position.set(L.bar.x - 0.012, L.bar.y - 0.067, z);
      drop.rotation.z = Math.PI * 0.18;
      this.bikeGroup.add(drop);
    }

    // Bottom bracket, chainring, cassette and chain.
    const bb = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.12, 20), alloy);
    bb.rotation.x = Math.PI / 2;
    bb.position.copy(L.bb);
    this.bikeGroup.add(bb);
    const chainRing = new THREE.Mesh(new THREE.TorusGeometry(0.095, 0.008, 10, 36), alloy);
    chainRing.position.copy(L.bb).add(v3(0, 0, -0.07));
    this.bikeGroup.add(chainRing);
    const cassette = new THREE.Mesh(new THREE.TorusGeometry(0.045, 0.008, 8, 28), alloy);
    cassette.position.copy(L.rearHub).add(v3(0, 0, -0.07));
    this.bikeGroup.add(cassette);
    mk(L.bb.clone().add(v3(0, 0.095, -0.07)), L.rearHub.clone().add(v3(0, 0.045, -0.07)), 0.004, alloy);
    mk(L.bb.clone().add(v3(0, -0.095, -0.07)), L.rearHub.clone().add(v3(0, -0.045, -0.07)), 0.004, alloy);

    // Wheels are in the frame's XY plane. The old model rotated the tire into
    // YZ while leaving the spokes in XY, which caused the crossed-wheel look.
    this.wheelSpin = [];
    [L.rearHub, L.frontHub].forEach((hub) => {
      const g = new THREE.Group();
      g.position.copy(hub);
      const tireMesh = new THREE.Mesh(new THREE.TorusGeometry(L.wheelR, 0.022, 12, 64), rubber);
      g.add(tireMesh);
      const rim = new THREE.Mesh(new THREE.TorusGeometry(L.wheelR - 0.035, 0.007, 8, 64), alloy);
      g.add(rim);
      for (let i = 0; i < 20; i++) {
        const angle = (i / 20) * Math.PI * 2;
        const endpoint = v3(
          Math.cos(angle) * (L.wheelR - 0.04),
          Math.sin(angle) * (L.wheelR - 0.04),
          i % 2 ? 0.018 : -0.018
        );
        const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.0017, 0.0017, 1, 6), alloy);
        setTube(sp, v3(0, 0, i % 2 ? -0.025 : 0.025), endpoint);
        g.add(sp);
      }
      const hubMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.11, 16), alloy);
      hubMesh.rotation.x = Math.PI / 2;
      g.add(hubMesh);
      this.bikeGroup.add(g);
      this.wheelSpin.push(g);
    });

    // Crank arms + pedals (animated)
    this.crankGroup = new THREE.Group();
    this.bikeGroup.add(this.crankGroup);
    this.crankL = this._addTube(this.crankGroup, 0.010, 0x69736e, 0.78);
    this.crankR = this._addTube(this.crankGroup, 0.010, 0x69736e, 0.78);
    this.pedalL = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.018, 0.065), carbonSoft);
    this.pedalR = new THREE.Mesh(new THREE.BoxGeometry(0.095, 0.018, 0.065), carbonSoft);
    this.crankGroup.add(this.pedalL);
    this.crankGroup.add(this.pedalR);
    this.bikeGroup.position.set(0, 0, 0);
    this.bikeGroup.rotation.set(0, 0, 0);
    this.bikeGroup.scale.setScalar(1);
    const floorY = geometry ? Math.min(L.rearHub.y, L.frontHub.y) - L.wheelR : 0;
    this.floor.position.y = floorY;
    this.grid.position.y = floorY + 0.002;
  }

  _buildHuman() {
    const skin = this._mat(0xe8c4a2, { roughness: 0.55, metalness: 0.05 });
    const jersey = this._mat(0x2f7fa8, { roughness: 0.48, metalness: 0.04 });
    const jerseyDark = this._mat(0x22516e, { roughness: 0.55, metalness: 0.03 });
    const shorts = this._mat(0x151d27, { roughness: 0.64, metalness: 0.03 });
    const shoe = this._mat(0x111714, { roughness: 0.48, metalness: 0.2 });

    // Layered head: cranium, jaw, ears, nose, visor and helmet.
    this.head = new THREE.Group();
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.11, 20, 18), skin);
    skull.scale.set(0.94, 1.06, 0.9);
    this.head.add(skull);
    const jaw = new THREE.Mesh(new THREE.SphereGeometry(0.085, 18, 14), skin);
    jaw.scale.set(0.92, 0.72, 0.88);
    jaw.position.set(0.015, -0.072, 0);
    this.head.add(jaw);
    for (const z of [-0.098, 0.098]) {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.018, 10, 8), skin);
      ear.position.set(-0.005, -0.005, z);
      this.head.add(ear);
    }
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.017, 0.048, 12), skin);
    nose.rotation.z = -Math.PI / 2;
    nose.position.set(0.108, -0.018, 0);
    this.head.add(nose);
    const helm = new THREE.Mesh(
      new THREE.SphereGeometry(0.126, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.57),
      this._mat(0xe9eef0, { roughness: 0.38, metalness: 0.18 })
    );
    helm.scale.set(1.02, 0.98, 1.0);
    helm.position.y = 0.018;
    this.head.add(helm);
    const visor = new THREE.Mesh(
      new THREE.BoxGeometry(0.052, 0.022, 0.13),
      this._mat(0x151d1a, { metalness: 0.3, roughness: 0.2 })
    );
    visor.position.set(0.092, 0.012, 0);
    this.head.add(visor);
    this.skelGroup.add(this.head);

    this.neck = this._addTaperedTube(this.skelGroup, 0.043, 0.036, skin);

    // Tapered chest + abdomen, with visible shoulder and pelvis width.
    this.torsoBaseHeight = 0.38;
    this.torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.12, 8, 20), jersey);
    this.skelGroup.add(this.torso);
    this.abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.13, 20, 14), jerseyDark);
    this.skelGroup.add(this.abdomen);
    this.pelvis = new THREE.Mesh(new THREE.SphereGeometry(0.12, 20, 14), shorts);
    this.pelvis.scale.set(1.35, 0.68, 0.84);
    this.skelGroup.add(this.pelvis);
    this.shoulderBridge = this._addTaperedTube(this.skelGroup, 0.043, 0.043, jersey);
    this.hipBridge = this._addTaperedTube(this.skelGroup, 0.042, 0.042, shorts);

    const jointSphere = (radius, material, scale = [1, 1, 1]) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 14), material);
      mesh.scale.set(...scale);
      this.skelGroup.add(mesh);
      return mesh;
    };
    this.shoulderL = jointSphere(0.053, jersey, [1, 0.92, 0.92]);
    this.shoulderR = jointSphere(0.053, jersey, [1, 0.92, 0.92]);
    this.elbowL = jointSphere(0.038, skin, [1, 0.94, 0.94]);
    this.elbowR = jointSphere(0.038, skin, [1, 0.94, 0.94]);
    this.hipL = jointSphere(0.057, shorts, [1, 0.96, 0.94]);
    this.hipR = jointSphere(0.057, shorts, [1, 0.96, 0.94]);
    this.kneeL = jointSphere(0.048, skin, [1, 0.92, 0.95]);
    this.kneeR = jointSphere(0.048, skin, [1, 0.92, 0.95]);

    // Anatomically tapered limb segments. Clothing follows cycling kit zones.
    this.limbMeshes = LIMBS.map((L) => {
      const material = L.kind === "upperArm"
        ? jersey
        : L.kind === "thigh"
          ? shorts
          : skin;
      const mesh = this._addTaperedTube(this.skelGroup, L.ra, L.rb, material);
      return { ...L, mesh };
    });

    this.handL = jointSphere(0.043, skin, [1.25, 0.66, 0.82]);
    this.handR = jointSphere(0.043, skin, [1.25, 0.66, 0.82]);
    this.footL = jointSphere(0.06, shoe, [1.55, 0.52, 0.72]);
    this.footR = jointSphere(0.06, shoe, [1.55, 0.52, 0.72]);

    this._loadRiggedHuman();
  }

  _loadRiggedHuman() {
    this.rigReady = false;
    const loader = new GLTFLoader();
    loader.load(
      "/static/models/cyclist.glb?v=20260816a",
      (gltf) => {
        const model = gltf.scene;
        const kit = this._mat(0x2a6f8f, { roughness: 0.55, metalness: 0.06 });
        const skin = this._mat(0xe0b089, { roughness: 0.62, metalness: 0.04 });
        model.traverse((object) => {
          if (object.isMesh || object.isSkinnedMesh) {
            object.castShadow = true;
            object.receiveShadow = true;
            const name = String(object.name || "").toLowerCase();
            object.material = /head|face|hand|skin/.test(name) ? skin : kit;
          }
        });
        this.characterGroup.add(model);
        this.rigModel = model;

        const tPose = (gltf.animations || []).find((clip) => clip.name === "TPose");
        if (tPose) {
          this.rigMixer = new THREE.AnimationMixer(model);
          const action = this.rigMixer.clipAction(tPose);
          action.play();
          this.rigMixer.setTime(0);
          action.paused = true;
        }

        const names = [
          "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
          "LeftArm", "LeftForeArm", "LeftHand",
          "RightArm", "RightForeArm", "RightHand",
          "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
          "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        ];
        const allBones = [];
        model.traverse((object) => {
          if (object.isBone) allBones.push(object);
        });
        const normalizedBoneName = (value) =>
          String(value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
        const aliases = {
          Hips: ["pelvis", "torso_joint_1"],
          Spine: ["spine_01", "torso_joint_2"], Spine1: ["spine_02", "torso_joint_2"], Spine2: ["spine_03", "torso_joint_3"],
          Neck: ["neck_01", "neck_joint_1"], Head: ["Head", "neck_joint_2"],
          LeftArm: ["upperarm_l", "arm_joint_L__4_"], LeftForeArm: ["lowerarm_l", "arm_joint_L__3_"], LeftHand: ["hand_l", "arm_joint_L__2_"],
          RightArm: ["upperarm_r", "arm_joint_R"], RightForeArm: ["lowerarm_r", "arm_joint_R__2_"], RightHand: ["hand_r", "arm_joint_R__3_"],
          LeftUpLeg: ["thigh_l", "leg_joint_L_1"], LeftLeg: ["calf_l", "leg_joint_L_2"], LeftFoot: ["foot_l", "leg_joint_L_3"], LeftToeBase: ["ball_l", "leg_joint_L_5"],
          RightUpLeg: ["thigh_r", "leg_joint_R_1"], RightLeg: ["calf_r", "leg_joint_R_2"], RightFoot: ["foot_r", "leg_joint_R_3"], RightToeBase: ["ball_r", "leg_joint_R_5"],
        };
        const findBone = (name) => {
          const candidates = [name, ...(aliases[name] || [])].map(normalizedBoneName);
          return allBones.find((bone) => {
            const boneName = normalizedBoneName(bone.name);
            return candidates.some((wanted) => boneName.endsWith(wanted));
          }) || null;
        };
        this.rigBones = {};
        for (const name of names) {
          this.rigBones[name] = findBone(name);
        }
        const required = [
          "Hips", "Head", "LeftArm", "LeftForeArm", "LeftHand",
          "RightArm", "RightForeArm", "RightHand",
          "LeftUpLeg", "LeftLeg", "LeftFoot",
          "RightUpLeg", "RightLeg", "RightFoot",
        ];
        if (required.some((name) => !this.rigBones[name])) {
          const missing = required.filter((name) => !this.rigBones[name]);
          console.warn(`Rigged rider is missing bones (${missing.join(", ")}); using fallback mannequin.`);
          this.characterGroup.remove(model);
          return;
        }

        this.rigRestQuaternions = new Map();
        this.rigRestPositions = new Map();
        model.traverse((object) => {
          if (object.isBone) {
            this.rigRestQuaternions.set(object, object.quaternion.clone());
            this.rigRestPositions.set(object, object.position.clone());
          }
        });
        this.characterGroup.position.set(0, 0, 0);
        this.characterGroup.quaternion.identity();
        this.characterGroup.scale.setScalar(1);
        model.updateMatrixWorld(true);

        const world = (bone) => bone.getWorldPosition(new THREE.Vector3());
        const restHipL = world(this.rigBones.LeftUpLeg);
        const restHipR = world(this.rigBones.RightUpLeg);
        const restShoL = world(this.rigBones.LeftArm);
        const restShoR = world(this.rigBones.RightArm);
        this.rigRestHip = restHipL.clone().add(restHipR).multiplyScalar(0.5);
        this.rigRestShoulder = restShoL.clone().add(restShoR).multiplyScalar(0.5);
        this.rigRestUp = this.rigRestShoulder.clone().sub(this.rigRestHip).normalize();
        this.rigRestSide = restHipL.clone().sub(restHipR).normalize();
        this.rigRestSide.addScaledVector(
          this.rigRestUp,
          -this.rigRestSide.dot(this.rigRestUp)
        ).normalize();
        this.rigRestForward = new THREE.Vector3()
          .crossVectors(this.rigRestSide, this.rigRestUp)
          .normalize();
        this.rigRestTorsoLength = Math.max(
          1e-5,
          this.rigRestShoulder.distanceTo(this.rigRestHip)
        );
        this.rigReady = true;
        this.skelGroup.visible = false;
        if (this.frames.length) this.showFrame(this.idx);
      },
      undefined,
      (error) => {
        console.warn("Rigged rider failed to load; using fallback mannequin.", error);
      }
    );
  }

  _alignRigBone(bone, child, targetStart, targetEnd) {
    if (!bone || !child) return;
    this.rigModel.updateMatrixWorld(true);
    const a = bone.getWorldPosition(new THREE.Vector3());
    const b = child.getWorldPosition(new THREE.Vector3());
    const current = b.sub(a);
    const desired = targetEnd.clone().sub(targetStart);
    if (current.lengthSq() < 1e-9 || desired.lengthSq() < 1e-9) return;
    current.normalize();
    desired.normalize();

    const delta = new THREE.Quaternion().setFromUnitVectors(current, desired);
    const worldQuaternion = bone.getWorldQuaternion(new THREE.Quaternion());
    const desiredWorld = delta.multiply(worldQuaternion);
    const parentWorld = bone.parent
      ? bone.parent.getWorldQuaternion(new THREE.Quaternion())
      : new THREE.Quaternion();
    bone.quaternion.copy(parentWorld.invert().multiply(desiredWorld)).normalize();
    bone.updateMatrixWorld(true);
  }

  _setRigBoneWorldPosition(bone, target) {
    if (!bone?.parent) return;
    bone.parent.updateMatrixWorld(true);
    const local = bone.parent.worldToLocal(target.clone());
    bone.position.copy(local);
    bone.updateMatrixWorld(true);
  }

  _fitAndAlignRigBone(bone, child, targetStart, targetEnd) {
    if (!bone || !child) return;
    this.rigModel.updateMatrixWorld(true);
    const currentA = bone.getWorldPosition(new THREE.Vector3());
    const currentB = child.getWorldPosition(new THREE.Vector3());
    const currentLength = currentA.distanceTo(currentB);
    const targetLength = targetStart.distanceTo(targetEnd);
    if (currentLength > 1e-7 && targetLength > 1e-7) {
      child.position.multiplyScalar(targetLength / currentLength);
      child.updateMatrixWorld(true);
    }
    this._alignRigBone(bone, child, targetStart, targetEnd);
  }

  _placeRiggedHuman(pts) {
    for (const [bone, quaternion] of this.rigRestQuaternions) {
      bone.quaternion.copy(quaternion);
      bone.position.copy(this.rigRestPositions.get(bone));
    }
    this.characterGroup.position.set(0, 0, 0);
    this.characterGroup.quaternion.identity();
    this.characterGroup.scale.setScalar(1);
    this.rigModel.updateMatrixWorld(true);

    const hipMid = pts[11].clone().add(pts[12]).multiplyScalar(0.5);
    const shoulderMid = pts[5].clone().add(pts[6]).multiplyScalar(0.5);
    const targetUp = shoulderMid.clone().sub(hipMid).normalize();
    // Bike +X is forward (rear hub → front hub). Do not infer facing from a
    // twisted hip line or the mesh will look sideways / backward.
    const targetForward = v3(1, 0, 0);
    targetForward.addScaledVector(targetUp, -targetForward.dot(targetUp)).normalize();
    const targetSide = new THREE.Vector3().crossVectors(targetUp, targetForward).normalize();
    const sourceBasis = new THREE.Matrix4().makeBasis(
      this.rigRestSide,
      this.rigRestUp,
      this.rigRestForward
    );
    const targetBasis = new THREE.Matrix4().makeBasis(targetSide, targetUp, targetForward);
    const rotationMatrix = targetBasis.multiply(sourceBasis.invert());
    this.characterGroup.quaternion.setFromRotationMatrix(rotationMatrix);
    const scale = shoulderMid.distanceTo(hipMid) / this.rigRestTorsoLength;
    this.characterGroup.scale.setScalar(scale);
    const transformedRestHip = this.rigRestHip
      .clone()
      .applyQuaternion(this.characterGroup.quaternion)
      .multiplyScalar(scale);
    this.characterGroup.position.copy(hipMid).sub(transformedRestHip);
    this.characterGroup.updateMatrixWorld(true);

    // Match the four chain roots exactly, then personalize every limb segment.
    this._setRigBoneWorldPosition(this.rigBones.LeftArm, pts[5]);
    this._setRigBoneWorldPosition(this.rigBones.RightArm, pts[6]);
    this._setRigBoneWorldPosition(this.rigBones.LeftUpLeg, pts[11]);
    this._setRigBoneWorldPosition(this.rigBones.RightUpLeg, pts[12]);

    this._fitAndAlignRigBone(this.rigBones.LeftArm, this.rigBones.LeftForeArm, pts[5], pts[7]);
    this._fitAndAlignRigBone(this.rigBones.LeftForeArm, this.rigBones.LeftHand, pts[7], pts[9]);
    this._fitAndAlignRigBone(this.rigBones.RightArm, this.rigBones.RightForeArm, pts[6], pts[8]);
    this._fitAndAlignRigBone(this.rigBones.RightForeArm, this.rigBones.RightHand, pts[8], pts[10]);
    this._fitAndAlignRigBone(this.rigBones.LeftUpLeg, this.rigBones.LeftLeg, pts[11], pts[13]);
    this._fitAndAlignRigBone(this.rigBones.LeftLeg, this.rigBones.LeftFoot, pts[13], pts[15]);
    this._fitAndAlignRigBone(this.rigBones.RightUpLeg, this.rigBones.RightLeg, pts[12], pts[14]);
    this._fitAndAlignRigBone(this.rigBones.RightLeg, this.rigBones.RightFoot, pts[14], pts[16]);

    // Preserve the character's anatomically authored neck/head relation. A
    // world-axis "upright head" correction twists rigs whose neck bones use a
    // different local basis and makes the rider appear to lean backwards.

    const footForward = v3(0.16, 0.01, 0);
    if (this.rigBones.LeftToeBase) {
      this._alignRigBone(
        this.rigBones.LeftFoot,
        this.rigBones.LeftToeBase,
        pts[15],
        pts[15].clone().add(footForward)
      );
    }
    if (this.rigBones.RightToeBase) {
      this._alignRigBone(
        this.rigBones.RightFoot,
        this.rigBones.RightToeBase,
        pts[16],
        pts[16].clone().add(footForward)
      );
    }
  }

  setFrames(frames, fps = 10, bikeGeometry = null, coordinateSystem = null) {
    this.frames = frames || [];
    this.fps = fps || 10;
    this.idx = 0;
    this._smooth = null;
    this.coordinateSystem = coordinateSystem || "motionbert-relative";
    this._buildBike(bikeGeometry || null);
    if (this.coordinateSystem === "bike-wheelbase" && this.frames.length) {
      this._frameBikeScene(bikeGeometry);
    }
    this.showFrame(0);
  }

  _frameBikeScene(bikeGeometry) {
    const xs = [];
    const ys = [];
    for (const frame of this.frames) {
      for (const p of frame.joints_xyz || []) {
        if (p && Number.isFinite(p[0]) && Number.isFinite(p[1])) {
          xs.push(p[0]);
          ys.push(p[1]);
        }
      }
    }
    const radius = Number(bikeGeometry?.wheel_radius || 0.32);
    for (const key of ["rear_hub", "front_hub"]) {
      const p = bikeGeometry?.[key];
      if (p) {
        xs.push(p[0] - radius, p[0] + radius);
        ys.push(p[1] - radius, p[1] + radius);
      }
    }
    if (!xs.length || !ys.length) return;
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const centerX = 0.5 * (minX + maxX);
    const centerY = 0.5 * (minY + maxY);
    const spanX = Math.max(0.8, maxX - minX);
    const spanY = Math.max(1.0, maxY - minY);
    const halfFov = THREE.MathUtils.degToRad(this.camera.fov * 0.5);
    const aspect = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
    const distanceY = spanY / (2 * Math.tan(halfFov));
    const distanceX = spanX / (2 * Math.tan(halfFov) * Math.max(aspect, 0.5));
    // Procedural skin/clothing extends beyond the joint centers; leave enough
    // head/helmet and shoe margin so the rider is never cropped.
    const distance = Math.max(distanceX, distanceY) * 1.38;
    this.controls.target.set(centerX, centerY, 0);
    // Video is shot from the rider's right (−Z). Stay on that side.
    this.camera.position.set(centerX + 0.08, centerY + 0.06, -distance);
    this.camera.lookAt(this.controls.target);
    this.controls.update();
  }

  _raw(joints) {
    return joints.map((p) => {
      if (!p || p.some((v) => v == null || Number.isNaN(v))) return null;
      return new THREE.Vector3(p[0], p[1], p[2]);
    });
  }

  /** Stabilize & remount into a readable side-view cyclist frame. */
  _prepare(raw) {
    const pts = raw.map((p) => (p ? p.clone() : null));
    if (!pts[11] || !pts[12] || !pts[5] || !pts[6]) return null;

    // Constrained output already lives in a stable wheelbase-normalized frame.
    // Do not re-root, yaw-rotate, depth-compress, or ground it again.
    if (this.coordinateSystem === "bike-wheelbase") return pts;

    const midHip = pts[11].clone().add(pts[12]).multiplyScalar(0.5);
    const midSho = pts[5].clone().add(pts[6]).multiplyScalar(0.5);
    const midWri =
      pts[9] && pts[10]
        ? pts[9].clone().add(pts[10]).multiplyScalar(0.5)
        : midSho.clone().add(v3(0.35, -0.05, 0));

    // Forward ≈ hip → hands on ground plane; fallback shoulder lean
    let fwd = midWri.clone().sub(midHip);
    fwd.y = 0;
    if (fwd.lengthSq() < 1e-5) {
      fwd = midSho.clone().sub(midHip);
      fwd.y = 0;
    }
    if (fwd.lengthSq() < 1e-5) fwd.set(1, 0, 0);
    fwd.normalize();
    const yaw = Math.atan2(fwd.z, fwd.x);
    const q = new THREE.Quaternion().setFromAxisAngle(v3(0, 1, 0), -yaw);

    for (const p of pts) {
      if (!p) continue;
      p.sub(midHip);
      p.applyQuaternion(q);
      // compress left-right noise so silhouette reads as a rider
      p.z *= 0.42;
    }

    // Re-anchor hips, put feet on ground plane
    const hip = pts[11].clone().add(pts[12]).multiplyScalar(0.5);
    let minY = Infinity;
    for (const p of pts) if (p) minY = Math.min(minY, p.y);
    const lift = -minY + 0.02; // ground y=0
    for (const p of pts) {
      if (!p) continue;
      p.y += lift;
      p.x -= hip.x;
      p.z -= hip.z * 0.15;
    }

    // Temporal smoothing for readability
    if (!this._smooth) {
      this._smooth = pts.map((p) => (p ? p.clone() : null));
    } else {
      const a = 0.35;
      for (let i = 0; i < pts.length; i++) {
        if (!pts[i] || !this._smooth[i]) {
          this._smooth[i] = pts[i] ? pts[i].clone() : null;
          continue;
        }
        this._smooth[i].lerp(pts[i], a);
        pts[i].copy(this._smooth[i]);
      }
    }
    return pts;
  }

  _placeHuman(pts) {
    if (this.rigReady) {
      this._placeRiggedHuman(pts);
      return;
    }
    const midSho = pts[5].clone().add(pts[6]).multiplyScalar(0.5);
    const midHip = pts[11].clone().add(pts[12]).multiplyScalar(0.5);
    const headPos = pts[0]
      ? pts[0].clone()
      : midSho.clone().add(v3(0.05, 0.22, 0));

    this.head.position.copy(headPos).add(v3(0, 0.025, 0));
    // Procedural head is authored facing +X (nose / visor). Keep that aligned
    // with the bicycle, with a slight downward look toward the bars.
    this.head.rotation.set(0, 0, -0.18);

    const neckTop = headPos.clone().add(v3(-0.01, -0.07, 0));
    setTube(this.neck, midSho, neckTop);

    const torsoDir = midSho.clone().sub(midHip);
    const torsoLen = Math.max(0.25, torsoDir.length());
    this.torso.scale.set(1.12, torsoLen / this.torsoBaseHeight, 0.68);
    this.torso.position.copy(midHip).add(midSho).multiplyScalar(0.5);
    this.torso.quaternion.setFromUnitVectors(v3(0, 1, 0), torsoDir.clone().normalize());

    this.abdomen.position.copy(midHip).lerp(midSho, 0.27);
    this.abdomen.scale.set(1.0, 0.82, 0.62);
    this.abdomen.quaternion.copy(this.torso.quaternion);
    this.pelvis.position.copy(midHip);
    this.pelvis.quaternion.copy(this.torso.quaternion);
    setTube(this.shoulderBridge, pts[5], pts[6]);
    setTube(this.hipBridge, pts[11], pts[12]);
    this.shoulderL.position.copy(pts[5]);
    this.shoulderR.position.copy(pts[6]);
    this.hipL.position.copy(pts[11]);
    this.hipR.position.copy(pts[12]);
    for (const [mesh, point] of [
      [this.elbowL, pts[7]], [this.elbowR, pts[8]],
      [this.kneeL, pts[13]], [this.kneeR, pts[14]],
    ]) {
      mesh.visible = !!point;
      if (point) mesh.position.copy(point);
    }

    for (const L of this.limbMeshes) {
      if (!pts[L.a] || !pts[L.b]) {
        L.mesh.visible = false;
        continue;
      }
      setTube(L.mesh, pts[L.a], pts[L.b]);
    }

    const hoodForward = v3(1, -0.12, 0).normalize();
    if (pts[9]) {
      this.handL.visible = true;
      this.handL.position.copy(pts[9]);
      this.handL.quaternion.setFromUnitVectors(v3(1, 0, 0), hoodForward);
    } else this.handL.visible = false;
    if (pts[10]) {
      this.handR.visible = true;
      this.handR.position.copy(pts[10]);
      this.handR.quaternion.setFromUnitVectors(v3(1, 0, 0), hoodForward);
    } else this.handR.visible = false;

    const shoeForward = v3(1, -0.08, 0).normalize();
    if (pts[15]) {
      this.footL.visible = true;
      this.footL.position.copy(pts[15]).add(v3(0.03, -0.01, 0));
      this.footL.quaternion.setFromUnitVectors(v3(1, 0, 0), shoeForward);
    }
    if (pts[16]) {
      this.footR.visible = true;
      this.footR.position.copy(pts[16]).add(v3(0.03, -0.01, 0));
      this.footR.quaternion.setFromUnitVectors(v3(1, 0, 0), shoeForward);
    }
  }

  _placeBike(pts) {
    const midHip = pts[11].clone().add(pts[12]).multiplyScalar(0.5);
    const midSho = pts[5].clone().add(pts[6]).multiplyScalar(0.5);
    const la = pts[15], ra = pts[16];

    const L = this.bikeLocal;
    if (!this.bikeIsCalibrated) {
      const torso = midSho.distanceTo(midHip);
      const scale = THREE.MathUtils.clamp(torso / 0.55, 0.75, 1.35);
      const seatWorld = midHip.clone();
      seatWorld.y -= 0.04 * scale;
      this.bikeGroup.scale.setScalar(scale);
      this.bikeGroup.rotation.set(0, 0, 0);
      this.bikeGroup.position.set(
        seatWorld.x - L.seat.x * scale,
        L.wheelR * scale - L.rearHub.y * scale,
        0
      );
    } else {
      this.bikeGroup.scale.setScalar(1);
      this.bikeGroup.rotation.set(0, 0, 0);
      this.bikeGroup.position.set(0, 0, 0);
    }
    this.bikeGroup.updateMatrixWorld(true);

    const crankLen = this._crankLen; // local units (pre-scale; parent scale applies)

    const placeCrank = (ankle, mesh, pedal, phase0) => {
      if (!ankle) {
        mesh.visible = false;
        pedal.visible = false;
        return 0;
      }
      mesh.visible = true;
      pedal.visible = true;
      const local = this.bikeGroup.worldToLocal(ankle.clone());
      let ang = Math.atan2(local.y - L.bb.y, local.x - L.bb.x);
      if (!Number.isFinite(ang)) ang = phase0;
      const tip = v3(
        L.bb.x + Math.cos(ang) * crankLen,
        L.bb.y + Math.sin(ang) * crankLen,
        local.z
      );
      setTube(mesh, L.bb, tip);
      pedal.position.copy(tip);
      return ang;
    };

    const a0 = placeCrank(la, this.crankL, this.pedalL, 0);
    placeCrank(ra, this.crankR, this.pedalR, a0 + Math.PI);

    const spin = (this.idx / Math.max(1, this.fps)) * 6.5;
    this.wheelSpin.forEach((w) => {
      w.rotation.z = -spin;
    });
  }

  showFrame(i) {
    if (!this.frames.length) return;
    this.idx = ((i % this.frames.length) + this.frames.length) % this.frames.length;
    const fr = this.frames[this.idx];
    if (!fr?.joints_xyz) return;

    const pts = this._prepare(this._raw(fr.joints_xyz));
    if (!pts) return;

    this._placeHuman(pts);
    this._placeBike(pts);
    if (this.onFrame) this.onFrame(this.idx);
  }

  resize() {
    const w = this.canvas.clientWidth || 640;
    const h = this.canvas.clientHeight || 400;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _tick(t) {
    requestAnimationFrame(this._tick);
    if (this.playing && this.frames.length) {
      if (t - this._last > 1000 / this.fps) {
        this._last = t;
        this.showFrame(this.idx + 1);
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
