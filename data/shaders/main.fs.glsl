// #version 460 core

// layout (location = 0) out vec4 oPixel;

// layout (location = 0) in vec2 iUvs;
// layout (location = 1) in flat uint iBaseColorIndex;

// layout (location = 2) uniform sampler2D[16] uTextures;

// struct ObjectData
// {
//     uint transformIndex;
//     uint baseColorIndex;
//     uint normalIndex;
// };

// layout (binding = 0) buffer BObjectData
// {
//     ObjectData[] objectData;
// };

// void main()
// {
//     oPixel = vec4(texture(uTextures[iBaseColorIndex], iUvs).rgb, 1.0);
// }

#version 330 core

out vec4 FragColor;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_scroll;
uniform vec3 u_camPos;
uniform vec3 u_camTarget;
uniform int u_flashlight;
uniform int u_renderMode;

const float MAX_STEPS = 500.0;
const float MIN_DIST_TO_SDF = 0.000001;
const float MAX_DIST_TO_TRAVEL = 100.0;
const float EPSILON = 0.001;
const float LOD_MULTIPLIER = 60;

float vmax(vec2 v) {
	return max(v.x, v.y);
}

float vmax(vec3 v) {
	return max(max(v.x, v.y), v.z);
}

float vmax(vec4 v) {
	return max(max(v.x, v.y), max(v.z, v.w));
}

struct Ray {
    vec3 origin;
    vec3 direction;
};

struct Light {
  float size;
  vec3 pos;
  vec3 col;
  vec3 dir;
  float focus;  
  float spread;
};

float fBox(vec3 p, vec3 b) {
	vec3 d = abs(p) - b;
	return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}

float fPlane(vec3 p, vec3 n, float distanceFromOrigin) {
	return dot(p, n) + distanceFromOrigin;
}

vec2 calcSDF(vec3 p) {
    float dist = 0.0;
    float matID = 0.0;


    float box = fBox(p, vec3(0.5));
    float plane = fPlane(p, vec3(0.0, 1.0, 0.0), 1.0);
    dist = min(box, plane);
    return vec2(dist, matID);
}

float calcAO(vec3 pos, vec3 normal) {
    float occ = 0.0;
    float sca = 1.0;

    for(int i=0; i<5; i++) {
        float hrconst = 0.03; // larger values = AO
        float hr = hrconst + 0.15*float(i)/4.0;
        vec3 aopos =  normal * hr + pos;
        float dd = calcSDF( aopos ).x;
        occ += (hr-dd)*sca;
        sca *= 0.95;
    }
    return clamp(1.0 - occ*1.5, 0.0, 1.0);
}

float calcDirLight(vec3 p, vec3 lookfrom, vec3 lookat, in float cut1, in float cut2) {
    vec3 lr = normalize(lookfrom - p);
    float intensity = dot(lr, normalize(lookfrom - lookat));
    return smoothstep(cut2, cut1, intensity);
}

float calcSoftshadowV3(in vec3 ro, in vec3 rd, float mint, float maxt, float w) {
    float res = 1.0;
    float ph = 1e20;
    float t = mint;
    for( int i=0; i<256 && t<maxt; i++ )
    {
        float h = calcSDF(ro + rd*t).x;
        if( h<0.001 )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, d/(w*max(0.0,t-y)) );
        ph = h;
        t += h;
    }
    return res;
}


vec3 calcLight(Light lightSource, vec3 pos, vec3 normal, vec3 rDirRef, float ambientOcc, vec3 material, float kSpecular) {
    float kDiffuse = 0.4,
        kAmbient = 0.005;

    vec3 iSpecular = 6.*lightSource.col,  // intensity
        iDiffuse = 2.*lightSource.col,
        iAmbient = 1.5*lightSource.col;

    float alpha_phong = 20.0; // phong alpha component


    vec3 lRay = normalize(lightSource.pos - pos);
    
    float light = calcDirLight(pos, lightSource.pos, lightSource.dir, cos(lightSource.focus), cos(lightSource.spread));
    vec3 lDirRef = reflect(lRay, normal);

    float shadow = 1.0;
    if (light > 0.001) { // no need to calculate shadow if we're in the dark
        shadow = calcSoftshadowV3(pos, lRay, 0.01, 3.0, lightSource.size);
    }
    vec3 dif = light*kDiffuse*iDiffuse*max(dot(lRay, normal), 0.)*shadow;
    vec3 spec = light*kSpecular*iSpecular*pow(max(dot(lRay, rDirRef), 0.), alpha_phong)*shadow;
    vec3 amb = light*kAmbient*iAmbient*ambientOcc;

    return material*(amb + dif + spec);
    
}

vec3 getNormal(vec3 p) {
    float d = calcSDF(p).x;

    vec2 e = vec2(EPSILON, 0.0);
    vec3 normal = d - vec3(
        calcSDF(p - e.xyy).x,
        calcSDF(p - e.yxy).x,
        calcSDF(p - e.yyx).x
    );
    return normalize(normal);
}

float raymarch(Ray ray) {
    float dOrig = 0.0; // distance from ray origin

    for(int i=0; i<MAX_STEPS; i++) {
        vec3 rPos = ray.origin + ray.direction * dOrig;
        float dSurf = calcSDF(rPos).x;
        dOrig += dSurf;
        if(dOrig > MAX_DIST_TO_TRAVEL || abs(dSurf) < MIN_DIST_TO_SDF*clamp(((dOrig*dOrig-3)*LOD_MULTIPLIER),1,MAX_DIST_TO_TRAVEL*MAX_DIST_TO_TRAVEL*LOD_MULTIPLIER)) break;
    }

    return dOrig;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
    vec3 camPos = vec3(0.0, 0.0, 3.0);
    vec3 camTarget = vec3(0.0, 0.0, 0.0);
    vec3 camDir = normalize(camTarget - camPos);
    vec3 camRight = normalize(cross(vec3(0.0, 1.0, 0.0), camDir));
    vec3 camUp = cross(camDir, camRight);
    vec3 rayDir = normalize(camDir + uv.x * camRight + uv.y * camUp);

    Ray ray = Ray(camPos, rayDir);
    float dist = raymarch(ray);

    if (dist < MAX_DIST_TO_TRAVEL) {
        vec3 p = ray.origin + dist * ray.direction;
        vec3 normal = getNormal(p);
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 color = vec3(1.0, 0.5, 0.2) * diff;
        FragColor = vec4(color, 1.0);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}