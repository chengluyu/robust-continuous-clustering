function [AO,C,l] = apply_ambient_occlusion(t,varargin)
  % APPLY_AMBIENT_OCCLUSION  Apply ambient occlusion to a given plot of a mesh.
  % Derive colors from the current plot and provide new modulated colors
  % according to the ambient occlusion computed for this mesh.
  %
  % AO = apply_ambient_occlusion(t)
  % [AO,C,l] = apply_ambient_occlusion(t,'ParameterName',parameter_value, ...)
  %
  % Inputs:
  %   t  handle to a `trisurf` or `patch` object
  %   Optional:
  %     'AO' followed by previously computed ambient occlusion values {[]}
  %     'AOFactor' followed by a scaler between [0,1] determining how much to
  %       to modulate by ambient occlusion {1}
  %     'AddLights' followed by whether to add lights {true}
  %     'SoftLighting' followed by whether to soft lighting {true}
  %     'Samples' followed by the number of samples to use when computing
  %       ambient occlusion {1000}
  %     'ColorMap' followed by a colormap to use
  %     'CAxis' followed by a caxis ("color axis") to use
  %     'Colors' followed by a #V|#F|1 list of colors to use
  %     'Unoriented' followed by whether to treat the surface as unoriented
  %       {false}
  % Outputs:
  %   AO  #V by 1 list of ambient occlusion values
  %   C  #V|#F by 3 modified colors
  %   l  #l list of light handles

  function [AO,C] = apply_ambient_occlusion_helper(t,AO,C,factor,unoriented)
    V = t.Vertices;
    Poly = t.Faces;
    % triangulate high order facets
    F = [];
    for c = 3:size(Poly,2)
      F = [F;Poly(:,[1 c-1 c])];
    end
    if isempty(C)
      C = t.FaceVertexCData;
    end
    if isnumeric(t.FaceColor)
      assert(numel(t.FaceColor) == 3);
      C = repmat(reshape(t.FaceColor,1,3),size(V,1),1);
    else
      if size(C,2) == 1
        CI = floor((C-CA(1))/(CA(2)-CA(1))*size(CM,1))+1;
        % maintain nans
        Cnan = isnan(C);
        C = squeeze(ind2rgb(CI,CM));
        C(Cnan,:) = nan;
      end
      if strcmp(t.FaceColor,'flat') && size(C,1) == size(V,1)
        C = C(F(:,1),:);
      end
    end
    if size(C,1) == size(V,1)
      t.FaceColor = 'interp';
      O = V;
      N = per_vertex_normals(V,F);
      % Matlab uses backwards normals
      t.VertexNormals =  -N;
      lighting phong;
    elseif size(C,1) == size(F,1)
      t.FaceColor = 'flat';
      % Could be fancy and use high-order quadrature.
      O = barycenter(V,F);
      N = normalizerow(normals(V,F));
      lighting flat;
    else
      error('Unknown number of colors');
    end
    if soft_lighting
      t.FaceLighting = 'phong';
      t.SpecularStrength = 0.1;
      t.DiffuseStrength = 0.1;
      % Boost ambient to maintain brightness
      t.AmbientStrength = 1.0;
    end
    if isempty(AO)
      AO = ambient_occlusion(V,F,O,N,samples);
      if unoriented
          AO = min(AO,ambient_occlusion(V,F,O,-N,samples));
      end
    end
    t.FaceVertexCData = bsxfun(@plus,(1-factor)*C,factor*bsxfun(@times,C,1-AO));
  end 

  AO = [];
  factor = 1;
  CM = [];
  CA = [];
  C = [];
  % default values
  add_lights = true;
  soft_lighting = true;
  samples = 1000;
  unoriented = false;
  % Map of parameter names to variable names
  params_to_variables = containers.Map( ...
    {'AO', 'AddLights','Factor','SoftLighting','Samples','ColorMap','CAxis','Colors','Unoriented'}, ...
    {'AO','add_lights','factor','soft_lighting','samples','CM','CA','C','unoriented'});
  v = 1;
  while v <= numel(varargin)
    param_name = varargin{v};
    if isKey(params_to_variables,param_name)
      assert(v+1<=numel(varargin));
      v = v+1;
      % Trick: use feval on anonymous function to use assignin to this workspace 
      feval(@()assignin('caller',params_to_variables(param_name),varargin{v}));
    else
      error('Unsupported parameter: %s',varargin{v});
    end
    v=v+1;
  end

  if isempty(CM)
    CM = colormap;
  end
  if isempty(CA)
    CA = caxis;
  end

  if ~exist('t','var') || isempty(t)
    c = get(gca,'Children');
    t = c(arrayfun(@(x) isa(x,'matlab.graphics.primitive.Patch'),c));
  end

  AOin = AO;
  Cin = C;
  for ii = 1:numel(t)
    if isempty(AOin)
      AOii = [];
    else
      if numel(t) == 1
        AOii = AOin;
      else
        AOii = AOin{ii};
      end
    end
    if isempty(Cin)
      Cii = [];
    else
      if numel(t) == 1
        Cii = Cin;
      else
        Cii = Cin{ii};
      end
    end

    tii = t(ii);
    [AOii,Cii] = apply_ambient_occlusion_helper(tii,AOii,Cii,factor,unoriented);
    if numel(t) == 1
      AO = AOii;
      C = Cii;
    else
      AO{ii} = AOii;
      C{ii} = Cii;
    end
  end

  if add_lights
    a = get(tii,'Parent');
    cen = [mean(a.XLim) mean(a.YLim) mean(a.ZLim)];
    l = {};
    l{1} = light('Position',[cen(1) cen(2) 10*(max(a.ZLim)-min(a.ZLim))+cen(3)],'Style','local','Color',[1 1 1]/3);
    l{2} = light('Position',[cen(1) 10*(max(a.ZLim)-min(a.ZLim))+cen(2) cen(3)],'Style','local','Color',[1 1 1]/3);
    l{3} = light('Position',[cen(1) 10*(min(a.ZLim)-max(a.ZLim))+cen(2) cen(3)],'Style','local','Color',[1 1 1]/3);
  end
end
