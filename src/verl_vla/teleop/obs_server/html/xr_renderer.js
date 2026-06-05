/*
Copyright 2026 Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

class TeleopXRRenderer {
  constructor() {
    this.canvas = null;
    this.gl = null;
    this.obsImage = null;
    this.obsTexture = null;
    this.obsTextureDirty = false;
    this.renderProgram = null;
    this.positionBuffer = null;
    this.texCoordBuffer = null;
  }

  setImage(dataUrl) {
    if (!this.obsImage) {
      this.obsImage = new Image();
      this.obsImage.onload = () => {
        this.obsTextureDirty = true;
      };
    }
    if (this.obsImage.src !== dataUrl) {
      this.obsImage.src = dataUrl;
    }
  }

  async prepareSession(session) {
    this.canvas = this.canvas || document.createElement("canvas");
    this.gl = this.canvas.getContext("webgl", {xrCompatible: true});
    if (!this.gl) {
      throw new Error("WebGL unavailable");
    }
    if (this.gl.makeXRCompatible) {
      await this.gl.makeXRCompatible();
    }
    session.updateRenderState({
      baseLayer: new XRWebGLLayer(session, this.gl)
    });
    this.initRenderer();
  }

  drawFrame(frame, session, referenceSpace) {
    if (!this.gl || !session || !session.renderState.baseLayer || !this.renderProgram) {
      return;
    }
    const gl = this.gl;
    const baseLayer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, baseLayer.framebuffer);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(0.02, 0.02, 0.02, 1.0);
    this.uploadTexture();

    const viewerPose = referenceSpace ? frame.getViewerPose(referenceSpace) : null;
    const views = viewerPose ? viewerPose.views : [null];
    gl.enable(gl.SCISSOR_TEST);
    for (const view of views) {
      const viewport = view
        ? baseLayer.getViewport(view)
        : {x: 0, y: 0, width: this.canvas.width, height: this.canvas.height};
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.scissor(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.clear(gl.COLOR_BUFFER_BIT);
      this.drawQuad();
    }
    gl.disable(gl.SCISSOR_TEST);
  }

  initRenderer() {
    const gl = this.gl;
    const vertexSource = `
      attribute vec2 a_position;
      attribute vec2 a_tex_coord;
      varying vec2 v_tex_coord;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_tex_coord = a_tex_coord;
      }
    `;
    const fragmentSource = `
      precision mediump float;
      varying vec2 v_tex_coord;
      uniform sampler2D u_image;
      void main() {
        gl_FragColor = texture2D(u_image, v_tex_coord);
      }
    `;
    this.renderProgram = this.createProgram(vertexSource, fragmentSource);
    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-0.42, 0.30, -0.42, -0.30, 0.42, 0.30, 0.42, -0.30]),
      gl.STATIC_DRAW
    );
    this.texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), gl.STATIC_DRAW);
    this.obsTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.obsTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([20, 20, 20, 255])
    );
  }

  uploadTexture() {
    const gl = this.gl;
    if (!gl || !this.obsTexture || !this.obsImage || !this.obsImage.complete || !this.obsTextureDirty) {
      return;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.obsTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.obsImage);
    this.obsTextureDirty = false;
  }

  drawQuad() {
    const gl = this.gl;
    gl.useProgram(this.renderProgram);

    const positionLocation = gl.getAttribLocation(this.renderProgram, "a_position");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const texCoordLocation = gl.getAttribLocation(this.renderProgram, "a_tex_coord");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texCoordBuffer);
    gl.enableVertexAttribArray(texCoordLocation);
    gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.obsTexture);
    gl.uniform1i(gl.getUniformLocation(this.renderProgram, "u_image"), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  createShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    const program = gl.createProgram();
    gl.attachShader(program, this.createShader(gl.VERTEX_SHADER, vertexSource));
    gl.attachShader(program, this.createShader(gl.FRAGMENT_SHADER, fragmentSource));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }
}
